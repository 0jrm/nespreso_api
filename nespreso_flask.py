import sys
sys.path.append("eoas_pyutils/")
import copy
import csv
from datetime import datetime
from nn_model import PredictionModel
from flask import Flask, request, send_file, jsonify
from werkzeug.exceptions import BadRequest
from functools import wraps
import os
import sys
import traceback
import pickle
import torch
import numpy as np
import xarray as xr
from singleFileModel_SAT import load_satellite_data, prepare_inputs
from TemperatureSalinityDataset import TemperatureSalinityDataset

app = Flask(__name__)

# Global variables for model, dataset, and device
model = None
model_checkpoint = None
device = None

# Define the path to the CSV log file
# CSV_LOG_FILE = '/var/www/virtualhosts/nespreso.coaps.fsu.edu/nespreso_api/nespreso_queries_log.csv'
CSV_LOG_FILE = './nespreso_queries_log.csv'

# Ensure the CSV file has a header row if it doesn't exist
if not os.path.exists(CSV_LOG_FILE):
    with open(CSV_LOG_FILE, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['timestamp', 'client_ip', 'latitudes', 'longitudes', 'dates', 'missing_data', 'status'])

def save_to_netcdf(pred_T, pred_S, depth, sss, sst, aviso, times, lat, lon, file_name='output.nc'):
    profile_number = np.arange(pred_T.shape[1])
    depth = depth.astype(np.float32)

    # Ensure that `times` is a numpy array of `numpy.datetime64`
    times = np.array([np.datetime64(time) for time in times])

    ds = xr.Dataset({
        'Temperature': (('depth', 'profile_number'), pred_T),
        'Salinity': (('depth', 'profile_number'), pred_S),
        'SSS': (('profile_number'), sss),
        'SST': (('profile_number'), sst),
        'AVISO': (('profile_number'), aviso),
        'time': (('profile_number'), times),
        'lat': (('profile_number'), lat),
        'lon': (('profile_number'), lon)
    }, coords={
        'profile_number': profile_number,
        'depth': depth
    })

    # Add units and attributes
    ds['Temperature'].attrs['units'] = 'Temperature (degrees Celsius)'
    ds['Salinity'].attrs['units'] = 'Salinity (practical salinity units)'
    ds['SSS'].attrs['units'] = 'Satellite sea surface salinity (psu)'
    ds['SST'].attrs['units'] = 'Satellite sea surface temperature (degrees Kelvin)'
    ds['AVISO'].attrs['units'] = 'Adjusted absolute dynamic topography (meters)'
    ds['lat'].attrs['units'] = 'Latitude'
    ds['lon'].attrs['units'] = 'Longitude'

    ds.attrs['description'] = 'Synthetic temperature and salinity profiles generated with NeSPReSO'
    ds.attrs['institution'] = 'COAPS, FSU'
    ds.attrs['author'] = 'Jose Roberto Miranda'
    ds.attrs['contact'] = 'jrm22n@fsu.edu'

    comp = dict(zlib=True, complevel=9)
    encoding = {var: comp for var in ds.data_vars}
    encoding.update({var: comp for var in ds.coords if var != 'profile_number'}) 

    ds.to_netcdf(file_name, encoding=encoding)

def load_model_and_dataset():

    global model,  device, model_checkpoint
    device = torch.device("cpu")
    print(f"Loading dataset and model to {device}")
    
    # Load model
    model_path = './data/model_0.8847_2024-10-09 20:45:20_sat.pth'
    model_checkpoint = torch.load(model_path, map_location=device)
    model_checkpoint = copy.deepcopy(model_checkpoint)
    model = PredictionModel(input_dim=9, layers_config=[512,512], output_dim=30, dropout_prob=0.2)
    model.load_state_dict(model_checkpoint["model_state_dict"])
    model.to(device)
    print("Model loaded successfully.")
    model.eval()

def update_log_status(csv_file, log_entry, status):
    """
    Updates the status for the given log entry in the CSV file.
    """
    with open(csv_file, mode='r') as file:
        lines = file.readlines()

    updated_lines = []
    for line in lines:
        if all(str(value) in line for value in log_entry[:6]):
            parts = line.strip().split(',')
            parts[-1] = status
            updated_lines.append(','.join(parts) + '\n')
        else:
            updated_lines.append(line)

    with open(csv_file, mode='w', newline='') as file:
        file.writelines(updated_lines)

def datetime_to_datenum(python_datetime):
    days_from_year_1_to_year_0 = 366
    matlab_base = datetime(1, 1, 1).toordinal() - days_from_year_1_to_year_0
    ordinal = python_datetime.toordinal()
    days_difference = ordinal - matlab_base

    hour, minute, second = python_datetime.hour, python_datetime.minute, python_datetime.second
    matlab_datenum = days_difference + (hour / 24.0) + (minute / 1440.0) + (second / 86400.0)
    return matlab_datenum

def validate_json(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not request.is_json:
            raise BadRequest(description="Request must be in JSON format")
        return f(*args, **kwargs)
    return decorated_function

@app.route("/predict", methods=["POST"])
@validate_json
def predict():
    try:
        data = request.get_json()

        # Extract and validate input data
        lat = data.get('lat')
        lon = data.get('lon')
        dates = data.get('date')

        if not lat or not lon or not dates:
            raise BadRequest(description="Missing 'lat', 'lon', or 'date' fields")

        if not (isinstance(lat, list) and isinstance(lon, list) and isinstance(dates, list)):
            raise BadRequest(description="'lat', 'lon', and 'date' must be lists")

        if len(lat) < 1 or len(lon) < 1 or len(dates) < 1:
            raise BadRequest(description="'lat', 'lon', and 'date' must contain at least one element")

        if not (len(lat) == len(lon) == len(dates)):
            raise BadRequest(description="Length of 'lat', 'lon', and 'date' must be equal")

        # Convert dates
        try:
            times = [datetime.strptime(date, "%Y-%m-%d") for date in dates]
        except ValueError:
            raise BadRequest(description="Dates must be in 'YYYY-MM-DD' format")

        lat = np.array(lat)
        lon = np.array(lon)

        print(f"Received request for {len(lat)} points.")

        # Prepare inputs and make predictions
        sss, sst, aviso = load_satellite_data(times, lat, lon)
        missing_data = np.max([np.sum(np.isnan(sss)), np.sum(np.isnan(sst)), np.sum(np.isnan(aviso))])
        dtime = [datetime_to_datenum(time) for time in times]
        input_data = prepare_inputs(dtime, lat, lon, sss, sst, aviso, model_checkpoint["input_params"])
        input_data = input_data.to(device)

        with torch.no_grad():
            pcs_predictions = model(input_data)
        pcs_predictions = pcs_predictions.cpu().numpy()

        pca_temp, pca_sal = model_checkpoint["pca_temp"], model_checkpoint["pca_sal"] #type class 'sklearn.decomposition._pca.PCA'

        pred_T = pca_temp.inverse_transform(pcs_predictions[:, :15]).T
        pred_S = pca_sal.inverse_transform(pcs_predictions[:, 15:]).T
        depth = np.arange(0, 1801)

        # Log request information to CSV with aggregated data
        log_entry = [
            datetime.utcnow().isoformat(),
            request.remote_addr,
            ';'.join(map(str, lat)),
            ';'.join(map(str, lon)),
            ';'.join([time.isoformat() for time in times]),
            str(missing_data),
            'Pending'
        ]
        
        with open(CSV_LOG_FILE, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(log_entry)

        # Attempt to generate and send the NetCDF file
        netcdf_file = f"/tmp/NeSPReSO_{dates[0]}_to_{dates[-1]}.nc"
        save_to_netcdf(pred_T, pred_S, depth, sss, sst, aviso, times, lat, lon, netcdf_file)

        # Update log entry to indicate success
        update_log_status(CSV_LOG_FILE, log_entry, 'Success')

        # Create the response object
        response = send_file(
            netcdf_file,
            mimetype='application/x-netcdf',
            as_attachment=True,
            download_name=f'NeSPReSO_{dates[0]}_to_{dates[-1]}.nc'
        )

        # Add custom headers
        response.headers["X-Missing-Data"] = str(missing_data)
        response.headers["X-Successful-Data"] = str(len(lat) - missing_data)

        return response

    except BadRequest as e:
        error_message = f"Bad Request: {e.description}"
        traceback_str = ''.join(traceback.format_tb(e.__traceback__))
        print(error_message)
        return jsonify({"error": error_message, "traceback": traceback_str}), 400

    except Exception as e:
        traceback_str = ''.join(traceback.format_tb(e.__traceback__))
        error_message = f"Error: {str(e)}\nTraceback: {traceback_str}"
        print(error_message)

        # Attempt to find and update the log entry
        try:
            update_log_status(CSV_LOG_FILE, log_entry, 'Failed')
        except Exception as log_exc:
            print(f"Failed to update log status: {log_exc}")

        return jsonify({"error": error_message}), 500

# Initialize the model and dataset at module load
load_model_and_dataset()

if __name__ == "__main__":
    app.run(debug=True)
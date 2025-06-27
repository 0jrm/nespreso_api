import csv
from datetime import datetime
from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from pydantic import BaseModel, conlist
import os
import sys
import traceback
import pickle
import torch
import numpy as np
import xarray as xr
from fastapi.responses import FileResponse
from singleFileModel_SAT import TemperatureSalinityDataset, PredictionModel, load_satellite_data, prepare_inputs
import time

# Add TemperatureSalinityDataset and PredictionModel to the global namespace, so it can run from bash
sys.modules['__main__'].TemperatureSalinityDataset = TemperatureSalinityDataset
sys.modules['__main__'].PredictionModel = PredictionModel
sys.modules['__mp_main__'].TemperatureSalinityDataset = TemperatureSalinityDataset
sys.modules['__mp_main__'].PredictionModel = PredictionModel

app = FastAPI()

# Global progress variable to track processing progress
progress = 0

# Define the path to the CSV log file
CSV_LOG_FILE = '/unity/g2/jmiranda/nespreso_api/log/nespreso_queries_log.csv'

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

    ds.to_netcdf(file_name, encoding=encoding, engine="netcdf4")

def load_model_and_dataset():
    device = torch.device("cpu")
    print(f"Loading dataset and model to {device}")
    
    # Load dataset
    dataset_pickle_file = '/unity/g2/jmiranda/SubsurfaceFields/GEM_SubsurfaceFields/config_dataset_full.pkl'
    if os.path.exists(dataset_pickle_file):
        with open(dataset_pickle_file, 'rb') as file:
            data = pickle.load(file)
            full_dataset = data['full_dataset']
    
    full_dataset.n_components = 15
    
    # Load model
    model_path = '/unity/g2/jmiranda/SubsurfaceFields/GEM_SubsurfaceFields/saved_models/model_Test Loss: 14.2710_2024-02-26 12:47:18_sat.pth'
    model = torch.load(model_path, map_location=device)
    model.to(device)
    model.eval()
    
    return model, full_dataset, device

@app.on_event("startup")
async def startup_event():
    global model, full_dataset, device
    model, full_dataset, device = load_model_and_dataset()

# Define the input validation schema using Pydantic
class PredictRequest(BaseModel):
    lat: conlist(float, min_length=1)
    lon: conlist(float, min_length=1)
    date: conlist(str, min_length=1)  # Expecting an array of dates in 'YYYY-MM-DD' format

@app.post("/predict")
async def predict(request: PredictRequest, req: Request, background_tasks: BackgroundTasks):
    try:
        # Validate and convert the dates
        times = [datetime.strptime(date, "%Y-%m-%d") for date in request.date]
        
        # Convert lat and lon lists to numpy arrays
        lat = np.array(request.lat)
        lon = np.array(request.lon)
        
        if len(lat) != len(lon) or len(lat) != len(times):
            raise HTTPException(status_code=400, detail="Length of lat, lon, and date must be equal")
        else:
            print(f"Received request for {len(lat)} points.")
            
        # Prepare inputs and make predictions
        sss, sst, aviso = load_satellite_data(times, lat, lon)
        missing_data = np.max([np.sum(np.isnan(sss)), np.sum(np.isnan(sst)), np.sum(np.isnan(aviso))])
        dtime = [datetime_to_datenum(time) for time in times]
        input_data = prepare_inputs(dtime, lat, lon, sss, sst, aviso, full_dataset.input_params)
        input_data = input_data.to(device)

        with torch.no_grad():
            pcs_predictions = model(input_data)
        pcs_predictions = pcs_predictions.cpu().numpy()
        synthetics = full_dataset.inverse_transform(pcs_predictions)

        pred_T = synthetics[0]
        pred_S = synthetics[1]
        depth = np.arange(full_dataset.min_depth, full_dataset.max_depth + 1)

        # Log request information to CSV with aggregated data
        log_entry = [
            datetime.utcnow().isoformat(),
            req.client.host,
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
        netcdf_file = f"/tmp/NeSPReSO_{request.date[0]}_to_{request.date[-1]}.nc"
        save_to_netcdf(pred_T, pred_S, depth, sss, sst, aviso, times, lat, lon, netcdf_file)

        # Update log entry to indicate success
        update_log_status(CSV_LOG_FILE, log_entry, 'Success')

        # Add task to delete the file after response is sent
        background_tasks.add_task(os.remove, netcdf_file)

        return FileResponse(
            netcdf_file, 
            media_type='application/x-netcdf', 
            filename=f'NeSPReSO_{request.date[0]}_to_{request.date[-1]}.nc',
            headers={
                "X-Missing-Data": str(missing_data),
                "X-Successful-Data": str(len(lat) - missing_data)
            }
        )

    except Exception as e:
        # Handle error (existing code)
        traceback_str = ''.join(traceback.format_tb(e.__traceback__))
        error_message = f"Error: {str(e)}\nTraceback: {traceback_str}"
        print(error_message)

        # Update log entry to indicate failure
        update_log_status(CSV_LOG_FILE, log_entry, 'Failed')

        raise HTTPException(status_code=500, detail=error_message)

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

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run("nespreso_host:app", host="0.0.0.0", port=8000, reload=True)

# $ uvicorn nespreso_host:app --host 0.0.0.0 --port 8000 --reload (for development, local only)
# $ gunicorn -w 2 -k uvicorn.workers.UvicornWorker nespreso_host:app --bind 0.0.0.0:8000

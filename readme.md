# NeSPReSO API and Client

## Overview

This project is a FastAPI-based service that generates NeSPReSO synthetic temperature and salinity profiles for specified latitude, longitude, and date inputs. The project includes both the API server and a client to interact with the API, making it easy to retrieve predictions either in JSON format or as a NetCDF file.

## Usage

### Getting Predictions Using the Client

The primary way to interact with the NeSPReSO service is through the provided client script (`nespreso_client.py`). The client sends requests to the API and retrieves predictions for the specified coordinates and dates.

```python
from nespreso_client import get_predictions

# Define your inputs
latitudes = [25.0, 26.0, 27.0]
longitudes = [-83.0, -84.0, -85.0]
dates = ["2010-08-20", "2018-08-21", "2018-08-22"]
output_file = "my_output.nc"

# Fetch predictions and save to a NetCDF file
result = get_predictions(latitudes, longitudes, dates, filename=output_file)
print("Result:", result)  # Should print the path to the saved NetCDF file
```

### Example Test Case Using Different Formats

The client can handle inputs in various formats, such as numpy arrays, pandas Series, and xarray DataArray. Here’s an example:

```python
import numpy as np
import pandas as pd
import xarray as xr
from nespreso_client import get_predictions

# Example input data in different formats
lat_np = np.array([45.0, 46.0, 47.0])
lon_pd = pd.Series([-30.0, -29.0, -28.0])
date_xr = xr.DataArray(pd.to_datetime(["2020-08-20", "2020-08-21", "2020-08-22"]))

# Fetch predictions
result = get_predictions(lat_np, lon_pd, date_xr, filename="output.nc")
print("NetCDF file saved as:", result)
```

### Running the FastAPI Server

To start the FastAPI server, which hosts the NeSPReSO service, use the following command:

```bash
uvicorn nespreso_host:app --host 0.0.0.0 --port 8000 --reload
```

Alternatively, for production, use Gunicorn with Uvicorn workers:

```bash
gunicorn -w 2 -k uvicorn.workers.UvicornWorker nespreso_host:app --bind 0.0.0.0:8000
```

## Files and Structure

### `nespreso_client.py`

This script acts as a client to interact with the FastAPI service, sending requests to the API and retrieving predictions.

- **`fetch_predictions(lat, lon, date, filename="output.nc", format="netcdf")`**: Sends an asynchronous request to the API and saves the response as a NetCDF file or returns JSON data.
- **`get_predictions(lat, lon, date, filename="output.nc")`**: A synchronous wrapper around `fetch_predictions`, simplifying usage in non-async environments.

### `nespreso_host.py`

This script defines the FastAPI-based web service. It loads a pre-trained model and dataset to generate synthetic temperature and salinity profiles based on the provided inputs.

- **Global Namespace Addition**: Registers `TemperatureSalinityDataset` and `PredictionModel` to ensure compatibility when running from bash.
- **`load_model_and_dataset()`**: Loads the NeSPReSO model and dataset required for generating predictions.
- **`save_to_netcdf()`**: Saves generated profiles to a NetCDF file.
- **`predict()`**: The primary endpoint (`/predict`) for generating and returning predictions in JSON or NetCDF format.
- **Logging**: Logs each query to a CSV file, recording client IP, input data, missing data points, and request status.

### `utils.py`

This module contains helper functions to preprocess the input data, ensuring it is in the correct format before being sent to the API.

- **`convert_to_numpy_array(data)`**: Converts input data to a numpy array.
- **`convert_to_list_of_floats(data)`**: Converts data to a list of floats.
- **`convert_date_to_iso_strings(date)`**: Converts date inputs to ISO 8601 strings (`'YYYY-MM-DD'`).
- **`preprocess_inputs(lat, lon, date)`**: Combines the above functions to prepare `lat`, `lon`, and `date` inputs for API requests.

## Nespresso-UI

This module contains the Nespresso-UI for visualizing map coordinates (latitude and longitude).

### Running the Nespresso-UI


To start the Nespresso-UI, navigate to the `nespresso-ui` directory and run one of the following commands:



```bash
npm install # to install necessary dependencies
npm start

```




## Dependencies

Ensure all dependencies are installed via `pip`:

```bash
pip install fastapi uvicorn httpx numpy pandas xarray torch scipy
```

## Notes

- **Model Paths**: Make sure the paths to the model and dataset files are correctly configured in `nespreso_host.py`.
- **Logging**: The API logs all requests, including input parameters and statuses, to a CSV file for tracking and debugging purposes.

## License

This project is licensed under the MIT License.
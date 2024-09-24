# NeSPReSO API and Client

## Overview

This project is a **Flask-based** service that generates NeSPReSO synthetic temperature and salinity profiles for specified latitude, longitude, and date inputs. The project includes both the API server and a client to interact with the API, making it easy to retrieve predictions as a NetCDF file.

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

### Running the Flask Server

To start the Flask server, which hosts the NeSPReSO service, use the following command:

```bash
python nespreso_flask.py
```

This will run the Flask app on the default port `5000`. You can access the API endpoint at `http://127.0.0.1:5000/predict`.

### Deployment with Apache and WSGI

For production deployment, you can integrate the Flask app with an Apache server using WSGI. Below are the instructions to set up the application with Apache using `mod_wsgi`.

#### `wsgi.py`

This file serves as the entry point for the WSGI server to run your Flask application.

```python
#!/usr/bin/env python3
import sys
import os
import logging

# Configure logging
logging.basicConfig(
    filename='/var/www/virtualhosts/nespreso.coaps.fsu.edu/nespreso_api/wsgi.log',
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s'
)

logging.info("================== Initializing Nespreso API ===========================")

# Add the project directory to the sys.path
project_home = '/var/www/virtualhosts/nespreso.coaps.fsu.edu/nespreso_api'
if project_home not in sys.path:
    sys.path.insert(0, project_home)
    logging.info(f"Added {project_home} to sys.path")

try:
    from nespreso_flask import app as application
    logging.info("WSGI application loaded successfully.")
except Exception as e:
    logging.exception("Failed to load WSGI application.")
    raise

logging.info("!!!!!!!!!! Done initializing WSGI application !!!!!!!!!!!!!!!!!")
```

**Note**: Update the `project_home` path to match the location where your project is stored.

#### Apache Configuration (`ozavala_custom_wsgi.conf`)

This configuration file tells Apache how to serve your Flask application using WSGI.

```
## nespreso_api
WSGIDaemonProcess nespreso_api user=apache group=apache python-home=/path/to/your/virtualenv
WSGIScriptAlias /nespreso_api /var/www/virtualhosts/nespreso.coaps.fsu.edu/nespreso_api/wsgi.py

<Directory /var/www/virtualhosts/nespreso.coaps.fsu.edu/nespreso_api>
    <Files wsgi.py>
        Require all granted
    </Files>
</Directory>

<Location /nespreso_api>
    WSGIProcessGroup nespreso_api
</Location>
```

- **`python-home`**: Update this to point to your virtual environment directory.
- **Paths**: Ensure all paths (`WSGIScriptAlias` and `<Directory>`) point to the correct locations of your `wsgi.py` file and project directory.

After updating the configuration file, enable the site and restart Apache:

```bash
sudo a2ensite ozavala_custom_wsgi.conf
sudo systemctl reload apache2
```

### Testing the Deployment

Once Apache is configured and running, you can test the API by sending a request to:

```
http://your_server_domain_or_ip/nespreso_api/predict
```

## Files and Structure

### `nespreso_client.py`

This script acts as a client to interact with the Flask service, sending requests to the API and retrieving predictions.

- **`fetch_predictions(lat, lon, date, filename="output.nc")`**: Sends an asynchronous request to the API and saves the response as a NetCDF file.
- **`get_predictions(lat, lon, date, filename="output.nc")`**: A synchronous wrapper around `fetch_predictions`, simplifying usage in non-async environments.

### `nespreso_flask.py`

This script defines the Flask-based web service. It loads a pre-trained model and dataset to generate synthetic temperature and salinity profiles based on the provided inputs.

- **`load_model_and_dataset()`**: Loads the NeSPReSO model and dataset required for generating predictions.
- **`save_to_netcdf()`**: Saves generated profiles to a NetCDF file.
- **`predict()`**: The primary endpoint (`/predict`) for generating and returning predictions as a NetCDF file.
- **Logging**: Logs each query to a CSV file, recording client IP, input data, missing data points, and request status.

### `wsgi.py`

The WSGI entry point script for deploying the Flask app with Apache.

### `ozavala_custom_wsgi.conf`

Apache configuration file for setting up the Flask app with WSGI.

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

Ensure all dependencies are installed via `conda`:

```bash
conda env create --prefix /conda/$USER/.conda/envs/nespreso_flask -f requirements.yml
```

## Notes

- **Model Paths**: Make sure the paths to the model and dataset files are correctly configured in `nespreso_flask.py`.
- **Logging**: The API logs all requests, including input parameters and statuses, to a CSV file for tracking and debugging purposes.
- **Permissions**: Ensure that the Apache user (`apache` or `www-data`) has the necessary permissions to read and execute files in your project directory.
- **Python Version**: Confirm that the Python version used in your virtual environment matches the one expected by Apache and `mod_wsgi`.

## License

This project is licensed under the MIT License.

```

**Additional Notes:**

- The `nespreso_flask.py` script should be updated to reflect any changes necessary for Flask (e.g., ensuring the app is named `app` so that `wsgi.py` can import it correctly).
- Make sure to adjust the paths in `wsgi.py` and `ozavala_custom_wsgi.conf` to match your server's directory structure.
- When deploying with Apache, double-check that all modules and dependencies are available to the Apache process, especially if using a virtual environment.

Feel free to include or adjust any other details specific to your deployment environment.
```

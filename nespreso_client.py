import httpx
import asyncio
from services.utils import preprocess_inputs
import datetime
import warnings
import xarray as xr

async def fetch_predictions(lat, lon, date, filename="output.nc", api_url=None):
    """
    Fetch predictions from the NeSPReSO Flask API (modular version).

    Parameters:
    - lat: list of float, list of latitudes
    - lon: list of float, list of longitudes
    - date: list of str, list of dates in 'YYYY-MM-DD' format
    - filename: str, path where the output NetCDF file will be saved (default is 'output.nc')
    - api_url: str, override the API URL (default: http://127.0.0.1:5000/v1/profile)

    Returns:
    - Saves the NetCDF file to `filename` and returns the file path.
    """
    # Default to new endpoint
    if api_url is None:
        api_url = "http://0.0.0.0:5000/v1/profile" # remote
        # api_url = "http://127.0.0.1:5000/v1/profile" # local
    if api_url.endswith("/predict"):
        warnings.warn("You are using the deprecated /predict endpoint. Please use /v1/profile.")

    data = {
        "lat": lat,
        "lon": lon,
        "date": date
    }
    timeout = httpx.Timeout(1800, connect=10.0)  # 30 minutes timeout for unlimited batch processing
    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post(api_url, json=data)
        if response.status_code == 200:
            content_type = response.headers.get('Content-Type')
            if content_type and content_type.startswith('application/x-netcdf'):
                with open(filename, "wb") as f:
                    f.write(response.content)
                print(f"NetCDF file saved as {filename}")
                return filename
            else:
                print("Unexpected content type:", content_type)
                return None
        else:
            print(f"Request failed with status code {response.status_code}")
            print("Response content:", response.content)
            return None

def get_predictions(lat, lon, date, filename="output.nc", api_url=None):
    """
    Synchronous wrapper for fetch_predictions.

    Parameters:
    - lat: list of float, list of latitudes
    - lon: list of float, list of longitudes
    - date: list of str, list of dates in 'YYYY-MM-DD' format
    - filename: str, path where the output NetCDF file will be saved (default is 'output.nc')
    - api_url: str, override the API URL (default: http://127.0.0.1:5000/v1/profile)

    Returns:
    - The result from fetch_predictions (NetCDF file path or None).
    """
    lat, lon, date = preprocess_inputs(lat, lon, date)
    print(f"Fetching predictions for {len(lat)} points...")
    if asyncio.get_event_loop().is_running():
        return asyncio.ensure_future(fetch_predictions(lat, lon, date, filename, api_url=api_url))
    else:
        return asyncio.run(fetch_predictions(lat, lon, date, filename, api_url=api_url))

# Example usage
if __name__ == "__main__":
    # latitudes = [25.0, 26.0, 27.0]
    # longitudes = [-83.0, -84.0, -85.0]
    # dates = ["2022-10-25", "2026-10-25", "2022-10-25"]
    import scipy
    mat = scipy.io.loadmat("/unity/g2/jmiranda/nespreso_api/uses/Idalia_profiles_Aug2Sep2023.mat")
    latitudes = mat['latar']
    longitudes = mat['lonar']
    dates = mat['timear']
    output_file = "/unity/g2/jmiranda/nespreso_api/uses/Idalia_profiles_Aug2Sep2023.nc"
    
    print(f"Loaded {len(latitudes)} points from MATLAB file")
    print(f"Latitude range: {latitudes.min():.4f} to {latitudes.max():.4f}")
    print(f"Longitude range: {longitudes.min():.4f} to {longitudes.max():.4f}")
    print(f"Date range: {dates.min()} to {dates.max()}")
    
    result = get_predictions(latitudes, longitudes, dates, filename=output_file)
    print("Result:", result)
    
    # Only try to read the file if it was successfully created
    if result and result != output_file:
        print("Warning: Expected output file path not returned")
    elif result:
        try:
            with xr.open_dataset(output_file) as ds:
                #temperature shape
                print(f'Temperature shape: {ds.Temperature.shape}')
                print(f'Temperature range: {ds.Temperature.min().values:.2f} to {ds.Temperature.max().values:.2f}')
                #salinity shape
                print(f'Salinity shape: {ds.Salinity.shape}')
                print(f'Salinity range: {ds.Salinity.min().values:.2f} to {ds.Salinity.max().values:.2f}')
        except FileNotFoundError:
            print(f"Output file {output_file} was not created")
        except Exception as e:
            print(f"Error reading output file: {e}")
    else:
        print("Failed to get predictions - output file not created")
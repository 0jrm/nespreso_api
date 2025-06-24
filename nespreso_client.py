import httpx
import asyncio
from utils import preprocess_inputs
import datetime
import warnings

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
    timeout = httpx.Timeout(5000, connect=10.0)
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
    latitudes = [25.0, 26.0, 27.0]
    longitudes = [-83.0, -84.0, -85.0]
    dates = ["2015-08-20", "2018-08-21", "2018-08-22"]
    output_file = "my_output.nc"
    result = get_predictions(latitudes, longitudes, dates, filename=output_file)
    print("Result:", result)
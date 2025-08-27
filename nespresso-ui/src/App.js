// src/App.js
import React, { useState } from 'react';
import MyMap from './components/Map';
import ErrorModal from './components/ErrorModal';

function App() {
  const [selectedFeatures, setSelectedFeatures] = useState({
    points: false,
    lines: false,
    areas: false,
  });
  const [mapData, setMapData] = useState({ latitudes: [], longitudes: [] });
  const [selectedCoords, setSelectedCoords] = useState(null);
  const defaultDate = new Date('2018-01-01');
  const [selectedDates, setSelectedDates] = useState([defaultDate]);
  const [errorModal, setErrorModal] = useState({ isOpen: false, message: '' });

  const handleCheckboxChange = (feature) => {
    setSelectedFeatures(prev => ({
      ...prev,
      [feature]: !prev[feature]
    }));
  };

  const handleSubmit = async () => {
    if (mapData.latitudes.length === 0 || mapData.longitudes.length === 0) {
      setErrorModal({
        isOpen: true,
        message: 'Please draw features on the map before submitting.'
      });
      return;
    }

    if (selectedDates.length === 0) {
      setErrorModal({
        isOpen: true,
        message: 'Please select at least one date before submitting.'
      });
      return;
    }

    // Format dates as YYYY-MM-DD strings for the API
    const formattedDates = selectedDates.map(date => date.toISOString().split('T')[0]);

    // Initialize arrays for the API request
    let requestLats = [];
    let requestLons = [];
    let requestDates = [];

    // Handle different feature types
    if (selectedFeatures.points) {
      // For points, use coordinates directly
      requestLats = [...mapData.latitudes];
      requestLons = [...mapData.longitudes];
      // Repeat dates for each point
      requestDates = formattedDates.reduce((acc, date) => {
        return [...acc, ...new Array(mapData.latitudes.length).fill(date)];
      }, []);
    }

    if (selectedFeatures.lines) {
      // For lines, interpolate points along the line
      for (let i = 0; i < mapData.latitudes.length - 1; i++) {
        const steps = 10; // Number of interpolation points between line vertices
        for (let j = 0; j <= steps; j++) {
          const fraction = j / steps;
          const lat = mapData.latitudes[i] + (mapData.latitudes[i + 1] - mapData.latitudes[i]) * fraction;
          const lon = mapData.longitudes[i] + (mapData.longitudes[i + 1] - mapData.longitudes[i]) * fraction;
          requestLats.push(lat);
          requestLons.push(lon);
        }
      }
      // Repeat dates for each interpolated point
      requestDates = formattedDates.reduce((acc, date) => {
        return [...acc, ...new Array(requestLats.length).fill(date)];
      }, []);
    }

    if (selectedFeatures.areas) {
      // For areas, create a grid of points within the polygon
      const minLat = Math.min(...mapData.latitudes);
      const maxLat = Math.max(...mapData.latitudes);
      const minLon = Math.min(...mapData.longitudes);
      const maxLon = Math.max(...mapData.longitudes);
      
      const gridSize = 0.1; // Grid spacing in degrees
      for (let lat = minLat; lat <= maxLat; lat += gridSize) {
        for (let lon = minLon; lon <= maxLon; lon += gridSize) {
          requestLats.push(lat);
          requestLons.push(lon);
        }
      }
      // Repeat dates for each grid point
      requestDates = formattedDates.reduce((acc, date) => {
        return [...acc, ...new Array(requestLats.length).fill(date)];
      }, []);
    }

    try {
      const response = await fetch('/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          lat: requestLats,
          lon: requestLons,
          date: requestDates
        }),
      });

      if (response.ok) {
        // Handle successful response
        const blob = await response.blob();
        
        // Get filename from Content-Disposition header or use default
        const contentDisposition = response.headers.get('Content-Disposition');
        const fileName = contentDisposition
          ? contentDisposition.split('filename=')[1].replace(/['"]/g, '')
          : 'NeSPReSO_output.nc';

        // Create a download link
        const url = window.URL.createObjectURL(
          new Blob([blob], { type: 'application/x-netcdf' })
        );
        
        // Create temporary link element and trigger download
        const link = document.createElement('a');
        link.href = url;
        link.setAttribute('download', fileName);
        document.body.appendChild(link);
        link.click();
        
        // Cleanup
        document.body.removeChild(link);
        window.URL.revokeObjectURL(url);

        // Show success message with statistics from headers
        const missingData = response.headers.get('X-Missing-Data') || '0';
        const successfulData = response.headers.get('X-Successful-Data') || '0';
        
        setErrorModal({
          isOpen: true,
          message: `Data downloaded successfully!\nSuccessful points: ${successfulData}\nMissing data points: ${missingData}`
        });
      } else {
        // Handle error response
        const errorData = await response.clone().json().catch(async () => {
          const text = await response.text();
          return { error: text };
        });
        
        let errorMessage = errorData.error;
        if (errorMessage && errorMessage.includes('<!DOCTYPE')) {
          errorMessage = `Server error (${response.status}): ${response.statusText}`;
        }
        
        throw new Error(errorMessage || 'Failed to submit data');
      }
    } catch (error) {
      setErrorModal({
        isOpen: true,
        message: `Error submitting data: ${error.message}`
      });
      console.error('Error details:', error);
    }
  };

  const handleMapClick = (coords) => {
    setSelectedCoords(coords);
  };

  const handleReset = () => {
    setSelectedFeatures({
      points: false,
      lines: false,
      areas: false,
    });
    setMapData({ latitudes: [], longitudes: [] });
    setSelectedCoords(null);
    setSelectedDates([defaultDate]);
    // If you have a ref to the Map component, you can call its reset method here
    // For example: mapRef.current.reset();
  };

  const handleDateChange = (event) => {
    const date = new Date(event.target.value);
    if (!date) return;
    
    // Check if date is already selected
    const dateExists = selectedDates.some(
      selectedDate => selectedDate.toDateString() === date.toDateString()
    );
    
    if (!dateExists) {
      setSelectedDates(prev => [...prev, date].sort((a, b) => a - b));
    }
  };

  const handleRemoveDate = (dateToRemove) => {
    setSelectedDates(prev => 
      prev.filter(date => date.toDateString() !== dateToRemove.toDateString())
    );
  };

  const handleCloseErrorModal = () => {
    setErrorModal({ isOpen: false, message: '' });
  };

  return (
    <div className="flex flex-col h-screen">
      <ErrorModal
        isOpen={errorModal.isOpen}
        message={errorModal.message}
        onClose={handleCloseErrorModal}
      />
      <header className="bg-blue-600 text-white p-4 shadow-md">
        <h1 className="text-2xl font-bold">Map Selector</h1>
      </header>
      <main className="flex flex-col md:flex-row flex-grow overflow-hidden">
        <div className="w-full md:w-1/2 p-4">
          <MyMap 
            onDataChange={setMapData} 
            selectedFeatures={selectedFeatures} 
            onMapClick={handleMapClick}
          />
        </div>
        <div className="w-full md:w-1/2 p-4 bg-gray-100 overflow-y-auto">
          <h2 className="text-xl font-semibold mb-4">Select Features</h2>
          <div className="space-y-2">
            <label className="flex items-center space-x-2">
              <input
                type="checkbox"
                checked={selectedFeatures.points}
                onChange={() => handleCheckboxChange('points')}
                className="form-checkbox"
              />
              <span>Points</span>
            </label>
            <label className="flex items-center space-x-2">
              <input
                type="checkbox"
                checked={selectedFeatures.lines}
                onChange={() => handleCheckboxChange('lines')}
                className="form-checkbox"
              />
              <span>Lines</span>
            </label>
            <label className="flex items-center space-x-2">
              <input
                type="checkbox"
                checked={selectedFeatures.areas}
                onChange={() => handleCheckboxChange('areas')}
                className="form-checkbox"
              />
              <span>Areas</span>
            </label>
          </div>
          <div className="mt-4 space-x-2">
            <button
              onClick={handleSubmit}
              className="bg-blue-500 hover:bg-blue-600 text-white font-bold py-2 px-4 rounded"
            >
              Submit
            </button>
            <button
              onClick={handleReset}
              className="bg-red-500 hover:bg-red-600 text-white font-bold py-2 px-4 rounded"
            >
              Reset
            </button>
          </div>
          {selectedCoords && (
            <div className="mt-4 p-3 bg-white rounded shadow">
              <h3 className="font-semibold mb-2">Selected Coordinates:</h3>
              <p>Latitude: {selectedCoords.lat.toFixed(4)}</p>
              <p>Longitude: {selectedCoords.lng.toFixed(4)}</p>
            </div>
          )}
          <div className="mt-4 p-3 bg-white rounded shadow">
            <h3 className="font-semibold mb-2">All Selected Points:</h3>
            <div className="max-h-60 overflow-y-auto">
              {mapData.latitudes.map((lat, index) => (
                <div key={index} className="mb-1">
                  <span className="font-medium">Point {index + 1}:</span> Lat: {lat.toFixed(4)}, Lng: {mapData.longitudes[index].toFixed(4)}
                </div>
              ))}
            </div>
          </div>
          <div className="mt-4 p-3 bg-white rounded shadow">
            <h3 className="font-semibold mb-2">Selected Dates:</h3>
            <div className="mb-4">
              <input
                type="date"
                onChange={handleDateChange}
                defaultValue="2018-01-01"
                min="2018-01-01"
                max={new Date().toISOString().split('T')[0]}
                className="border rounded p-2 w-full"
              />
            </div>
            <div className="max-h-60 overflow-y-auto">
              {selectedDates.map((date, index) => (
                <div key={index} className="flex justify-between items-center mb-1 p-2 bg-gray-50 rounded">
                  <span>{date.toISOString().split('T')[0]}</span>
                  <button
                    onClick={() => handleRemoveDate(date)}
                    className="text-red-500 hover:text-red-700"
                  >
                    ×
                  </button>
                </div>
              ))}
            </div>
          </div>
        </div>
      </main>
      <footer className="bg-gray-800 text-white p-4 text-center">
        <p>&copy; {new Date().getFullYear()} Your Company</p>
      </footer>
    </div>
  );
}

export default App;

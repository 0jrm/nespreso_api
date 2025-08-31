// src/components/Map.js
import React, { useState, useEffect } from 'react';
import { MapContainer, TileLayer, Marker, Popup, Polyline, Polygon, useMapEvents } from 'react-leaflet';
import 'leaflet/dist/leaflet.css';
import L from '../utils/leafleticon';
import Papa from 'papaparse';

// Gulf of Mexico coordinates (approximate center)
const GULF_OF_MEXICO_CENTER = [25.5, -91.0];
const INITIAL_ZOOM = 6;

const MapEventHandler = ({ addMarker, addPolyline, addPolygon, onMapClick, selectedFeatures }) => {
  const [polylinePoints, setPolylinePoints] = useState([]);
  const [polygonPoints, setPolygonPoints] = useState([]);

  useMapEvents({
    click(e) {
      const { lat, lng } = e.latlng;
      onMapClick({ lat, lng });

      if (selectedFeatures.points) {
        addMarker({ lat, lng });
      }

      if (selectedFeatures.lines) {
        addMarker({ lat, lng });
        setPolylinePoints([...polylinePoints, [lat, lng]]);
        if (polylinePoints.length === 1) {
          addPolyline([...polylinePoints, [lat, lng]]);
          setPolylinePoints([]);
        }
      }

      if (selectedFeatures.areas) {
        addMarker({ lat, lng });
        setPolygonPoints([...polygonPoints, [lat, lng]]);
        if (polygonPoints.length === 2) {
          addPolygon([...polygonPoints, [lat, lng]]);
          setPolygonPoints([]);
        }
      }
    },
  });

  return null;
};

const MyMap = ({ onDataChange, selectedFeatures, onMapClick }) => {
  const [markers, setMarkers] = useState([]);
  const [polylines, setPolylines] = useState([]);
  const [polygons, setPolygons] = useState([]);
  const [boundaryCoordinates, setBoundaryCoordinates] = useState([]);

  useEffect(() => {
    // Fetch and parse the CSV file using PapaParse
    fetch('/coord.csv')
      .then(response => {
        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }
        return response.text();
      })
      .then(csvText => {
        console.log("CSV File Content:", csvText); // Debug: Log CSV content

        Papa.parse(csvText, {
          header: true, // Treat the first row as headers
          skipEmptyLines: true,
          complete: (results) => {
            console.log("Parsed Results:", results); // Debug: Log parsed results

            const parsedCoordinates = results.data.map(row => {
              const latitude = parseFloat(row.Latitude); // Note the uppercase 'L'
              const longitude = parseFloat(row.Longitude); // Note the uppercase 'L'
              return [latitude, longitude];
            }).filter(coord => !isNaN(coord[0]) && !isNaN(coord[1])); // Filter out invalid entries

            console.log("Boundary Coordinates:", parsedCoordinates); // Debug: Log boundary coordinates
            setBoundaryCoordinates(parsedCoordinates);
          },
          error: (error) => {
            console.error("Error parsing CSV:", error);
          }
        });
      })
      .catch(error => console.error('Error fetching the CSV file:', error));
  }, []);

  const addMarker = (marker) => {
    setMarkers([...markers, marker]);
  };

  const addPolyline = (polyline) => {
    setPolylines([...polylines, polyline]);
  };

  const addPolygon = (polygon) => {
    setPolygons([...polygons, polygon]);
  };

  useEffect(() => {
    const latitudes = markers.map(marker => marker.lat);
    const longitudes = markers.map(marker => marker.lng);

    // Include polyline and polygon points
    polylines.forEach(polyline => {
      polyline.forEach(point => {
        latitudes.push(point[0]);
        longitudes.push(point[1]);
      });
    });

    polygons.forEach(polygon => {
      polygon.forEach(point => {
        latitudes.push(point[0]);
        longitudes.push(point[1]);
      });
    });

    onDataChange({ latitudes, longitudes });
  }, [markers, polylines, polygons, onDataChange]);

  return (
    <MapContainer center={GULF_OF_MEXICO_CENTER} zoom={INITIAL_ZOOM} className="h-full w-full">
      <TileLayer
        url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
        attribution='&copy; <a href="http://osm.org/copyright">OpenStreetMap</a> contributors'
      />
      <MapEventHandler 
        addMarker={addMarker} 
        addPolyline={addPolyline} 
        addPolygon={addPolygon} 
        onMapClick={onMapClick}
        selectedFeatures={selectedFeatures}
      />
      {markers.map((marker, idx) => (
        <Marker key={idx} position={[marker.lat, marker.lng]} icon={new L.Icon.Default()}>
          <Popup>Coordinates: {marker.lat.toFixed(4)}, {marker.lng.toFixed(4)}</Popup>
        </Marker>
      ))}
      {polylines.map((polyline, idx) => (
        <Polyline key={idx} positions={polyline} color="blue" />
      ))}
      {polygons.map((polygon, idx) => (
        <Polygon key={idx} positions={polygon} color="green" />
      ))}

      {/* Add the shaded region */}
      <Polygon
        positions={boundaryCoordinates}
        pathOptions={{ fillColor: 'purple', fillOpacity: 0.3, color: 'purple', weight: 2 }}
      />
    </MapContainer>
  );
};

export default MyMap;

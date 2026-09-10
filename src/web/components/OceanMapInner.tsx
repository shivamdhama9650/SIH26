"use client";

import React, { useEffect, useState } from "react";
import { MapContainer, TileLayer, Marker, Popup, Rectangle, useMapEvents } from "react-leaflet";
import L from "leaflet";
import "leaflet/dist/leaflet.css";

// Fix standard Leaflet marker icon in Next.js/bundler environments
const markerIcon = new L.Icon({
  iconUrl: "https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon.png",
  iconRetinaUrl: "https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon-2x.png",
  shadowUrl: "https://unpkg.com/leaflet@1.9.4/dist/images/marker-shadow.png",
  iconSize: [25, 41],
  iconAnchor: [12, 41],
  popupAnchor: [1, -34],
  shadowSize: [41, 41],
});

interface OceanMapInnerProps {
  latitude: number;
  longitude: number;
  onLocationSelect?: (lat: number, lon: number) => void;
  latMin?: number;
  latMax?: number;
  lonMin?: number;
  lonMax?: number;
}

function MapClickHandler({
  onLocationSelect,
  latMin,
  latMax,
  lonMin,
  lonMax,
}: {
  onLocationSelect?: (lat: number, lon: number) => void;
  latMin: number;
  latMax: number;
  lonMin: number;
  lonMax: number;
}) {
  useMapEvents({
    click(e) {
      if (!onLocationSelect) return;
      const clickedLat = parseFloat(e.latlng.lat.toFixed(2));
      const clickedLon = parseFloat(e.latlng.lng.toFixed(2));
      // Only trigger if click falls within valid domain
      if (clickedLat >= latMin && clickedLat <= latMax && clickedLon >= lonMin && clickedLon <= lonMax) {
        onLocationSelect(clickedLat, clickedLon);
      }
    },
  });
  return null;
}

export const OceanMapInner: React.FC<OceanMapInnerProps> = ({
  latitude,
  longitude,
  onLocationSelect,
  latMin = 5.0,
  latMax = 30.0,
  lonMin = 45.0,
  lonMax = 105.0,
}) => {
  // Domain bounding box: [[latMin, lonMin], [latMax, lonMax]]
  const bounds: [[number, number], [number, number]] = [
    [latMin, lonMin],
    [latMax, lonMax],
  ];

  return (
    <MapContainer
      center={[latitude, longitude]}
      zoom={4}
      minZoom={3}
      maxZoom={8}
      scrollWheelZoom={false}
      style={{ height: "100%", width: "100%", borderRadius: "0.5rem" }}
    >
      <TileLayer
        attribution='Tiles &copy; Esri &mdash; Source: Esri, DeLorme, NAVTEQ, USGS, Intermap, iPC, NRCAN, Esri Japan, METI, Esri China (Hong Kong), Esri (Thailand), TomTom'
        url="https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}"
      />

      {/* Target Domain Boundary Box */}
      <Rectangle
        bounds={bounds}
        pathOptions={{
          color: "#0284c7",
          weight: 2,
          fillColor: "#0284c7",
          fillOpacity: 0.08,
          dashArray: "6, 6",
        }}
      />

      {/* Selected Location Marker */}
      <Marker position={[latitude, longitude]} icon={markerIcon}>
        <Popup>
          <div className="text-xs font-sans">
            <p className="font-semibold text-slate-800">Target Station</p>
            <p className="text-slate-600 font-mono mt-0.5">
              Lat: {latitude.toFixed(2)}°N<br />
              Lon: {longitude.toFixed(2)}°E
            </p>
          </div>
        </Popup>
      </Marker>

      <MapClickHandler
        onLocationSelect={onLocationSelect}
        latMin={latMin}
        latMax={latMax}
        lonMin={lonMin}
        lonMax={lonMax}
      />
    </MapContainer>
  );
};

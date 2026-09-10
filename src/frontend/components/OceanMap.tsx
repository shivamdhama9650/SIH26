"use client";

import React, { useState, useEffect } from "react";
import dynamic from "next/dynamic";
import { Map, MapPin, Navigation, Info } from "lucide-react";
import { ConfigResponse } from "@/lib/api";

// Dynamically import Leaflet with SSR disabled
const DynamicMapInner = dynamic(
  () => import("./OceanMapInner").then((mod) => mod.OceanMapInner),
  {
    ssr: false,
    loading: () => (
      <div className="w-full h-full flex flex-col items-center justify-center bg-slate-100 rounded-lg text-slate-400">
        <Map className="w-8 h-8 animate-pulse text-cyan-600 mb-2" />
        <span className="text-xs font-medium">Initializing Oceanographic Basemap...</span>
      </div>
    ),
  }
);

interface OceanMapProps {
  latitude: number;
  longitude: number;
  config: ConfigResponse | null;
  onLocationSelect?: (lat: number, lon: number) => void;
}

export const OceanMap: React.FC<OceanMapProps> = ({
  latitude,
  longitude,
  config,
  onLocationSelect,
}) => {
  const [isClient, setIsClient] = useState(false);

  useEffect(() => {
    setIsClient(true);
  }, []);

  const latMin = config?.latitude_min ?? 5.0;
  const latMax = config?.latitude_max ?? 30.0;
  const lonMin = config?.longitude_min ?? 45.0;
  const lonMax = config?.longitude_max ?? 105.0;

  return (
    <div className="bg-white border border-slate-200 rounded-xl p-5 shadow-sm space-y-3 flex flex-col h-full">
      {/* Card Header */}
      <div className="flex items-center justify-between pb-3 border-b border-slate-100">
        <div className="flex items-center space-x-2">
          <div className="p-1.5 rounded bg-blue-50 text-blue-700">
            <Map className="w-4 h-4" />
          </div>
          <div>
            <h2 className="text-sm font-semibold uppercase tracking-wider text-slate-800">
              North Indian Ocean Domain
            </h2>
          </div>
        </div>
        <div className="flex items-center space-x-1.5 text-xs text-slate-500 bg-slate-50 px-2.5 py-1 rounded-md border border-slate-200 font-mono">
          <MapPin className="w-3.5 h-3.5 text-red-500" />
          <span>{latitude.toFixed(2)}°N, {longitude.toFixed(2)}°E</span>
        </div>
      </div>

      {/* Map Display */}
      <div className="relative flex-1 min-h-[300px] w-full rounded-lg overflow-hidden border border-slate-200 bg-slate-50">
        {isClient ? (
          <DynamicMapInner
            latitude={latitude}
            longitude={longitude}
            onLocationSelect={onLocationSelect}
            latMin={latMin}
            latMax={latMax}
            lonMin={lonMin}
            lonMax={lonMax}
          />
        ) : (
          <div className="w-full h-full flex items-center justify-center text-slate-400">
            <span className="text-xs">Loading Domain Map...</span>
          </div>
        )}

        {/* Legend Overlay */}
        <div className="absolute bottom-2 left-2 z-20 bg-white/90 backdrop-blur-xs border border-slate-200 px-2 py-1.5 rounded-md text-[10px] text-slate-700 shadow-sm space-y-0.5 pointer-events-none">
          <div className="flex items-center space-x-1.5">
            <span className="w-3 h-0.5 border-b-2 border-dashed border-cyan-600 inline-block" />
            <span className="font-semibold">Domain:</span>
            <span>5°N–30°N, 45°E–105°E</span>
          </div>
          <div className="flex items-center space-x-1.5">
            <span className="w-2 h-2 rounded-full bg-red-500 inline-block" />
            <span>Click ocean to reposition marker</span>
          </div>
        </div>
      </div>

      <p className="text-[11px] text-slate-400">
        Target domain covers the Arabian Sea, Bay of Bengal, and the Northern Equatorial Indian Ocean basin.
      </p>
    </div>
  );
};

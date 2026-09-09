"use client";

import React, { useState } from "react";
import { Compass, Calendar, Play, Loader2, MapPin, Sparkles } from "lucide-react";
import { ConfigResponse, SamplePoint } from "@/lib/api";

interface PredictionFormProps {
  config: ConfigResponse | null;
  samplePoints: SamplePoint[];
  latitude: number;
  longitude: number;
  date: string;
  onLatitudeChange: (val: number) => void;
  onLongitudeChange: (val: number) => void;
  onDateChange: (val: string) => void;
  onSubmit: () => void;
  isLoading: boolean;
}

export const PredictionForm: React.FC<PredictionFormProps> = ({
  config,
  samplePoints,
  latitude,
  longitude,
  date,
  onLatitudeChange,
  onLongitudeChange,
  onDateChange,
  onSubmit,
  isLoading,
}) => {
  const [validationError, setValidationError] = useState<string | null>(null);

  const minLat = config?.latitude_min ?? 5.0;
  const maxLat = config?.latitude_max ?? 30.0;
  const minLon = config?.longitude_min ?? 45.0;
  const maxLon = config?.longitude_max ?? 105.0;

  const handleFormSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (latitude < minLat || latitude > maxLat) {
      setValidationError(`Latitude must be between ${minLat}°N and ${maxLat}°N.`);
      return;
    }
    if (longitude < minLon || longitude > maxLon) {
      setValidationError(`Longitude must be between ${minLon}°E and ${maxLon}°E.`);
      return;
    }
    setValidationError(null);
    onSubmit();
  };

  const selectSamplePoint = (pt: SamplePoint) => {
    onLatitudeChange(pt.latitude);
    onLongitudeChange(pt.longitude);
    setValidationError(null);
  };

  return (
    <div className="bg-white border border-slate-200 rounded-xl p-5 shadow-sm space-y-4">
      <div className="flex items-center justify-between pb-3 border-b border-slate-100">
        <div className="flex items-center space-x-2">
          <div className="p-1.5 rounded bg-cyan-50 text-cyan-700">
            <Compass className="w-4 h-4" />
          </div>
          <h2 className="text-sm font-semibold uppercase tracking-wider text-slate-800">
            Prediction Controls
          </h2>
        </div>
        <span className="text-[11px] text-slate-500 font-mono">
          3×3 Grid Centered
        </span>
      </div>

      {validationError && (
        <div className="p-2.5 bg-red-50 border border-red-200 text-red-700 text-xs rounded-lg">
          {validationError}
        </div>
      )}

      <form onSubmit={handleFormSubmit} className="space-y-4">
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
          {/* Latitude */}
          <div>
            <label className="block text-xs font-semibold text-slate-700 mb-1">
              Latitude (°N)
            </label>
            <div className="relative">
              <input
                id="latitude-input"
                type="number"
                step="0.01"
                min={minLat}
                max={maxLat}
                value={latitude}
                onChange={(e) => {
                  onLatitudeChange(parseFloat(e.target.value) || 0);
                  setValidationError(null);
                }}
                disabled={isLoading}
                className="w-full px-3 py-2 bg-slate-50 border border-slate-300 rounded-lg text-sm text-slate-900 focus:outline-none focus:ring-2 focus:ring-cyan-500 focus:bg-white font-mono transition-all"
                required
              />
              <span className="absolute right-3 top-2.5 text-xs text-slate-400 font-mono">
                5°–30°N
              </span>
            </div>
          </div>

          {/* Longitude */}
          <div>
            <label className="block text-xs font-semibold text-slate-700 mb-1">
              Longitude (°E)
            </label>
            <div className="relative">
              <input
                id="longitude-input"
                type="number"
                step="0.01"
                min={minLon}
                max={maxLon}
                value={longitude}
                onChange={(e) => {
                  onLongitudeChange(parseFloat(e.target.value) || 0);
                  setValidationError(null);
                }}
                disabled={isLoading}
                className="w-full px-3 py-2 bg-slate-50 border border-slate-300 rounded-lg text-sm text-slate-900 focus:outline-none focus:ring-2 focus:ring-cyan-500 focus:bg-white font-mono transition-all"
                required
              />
              <span className="absolute right-3 top-2.5 text-xs text-slate-400 font-mono">
                45°–105°E
              </span>
            </div>
          </div>
        </div>

        {/* Date */}
        <div>
          <label className="block text-xs font-semibold text-slate-700 mb-1">
            Observation Date
          </label>
          <div className="relative">
            <input
              id="date-input"
              type="date"
              value={date}
              onChange={(e) => onDateChange(e.target.value)}
              disabled={isLoading}
              className="w-full px-3 py-2 bg-slate-50 border border-slate-300 rounded-lg text-sm text-slate-900 focus:outline-none focus:ring-2 focus:ring-cyan-500 focus:bg-white font-mono transition-all"
              required
            />
            <Calendar className="absolute right-3 top-2.5 w-4 h-4 text-slate-400 pointer-events-none" />
          </div>
        </div>

        {/* Preset Sample Points */}
        {samplePoints && samplePoints.length > 0 && (
          <div>
            <div className="flex items-center space-x-1.5 mb-2">
              <Sparkles className="w-3.5 h-3.5 text-cyan-600" />
              <label className="text-xs font-medium text-slate-600">
                Preset Oceanographic Stations:
              </label>
            </div>
            <div className="flex flex-wrap gap-1.5">
              {samplePoints.map((pt) => (
                <button
                  key={pt.name}
                  type="button"
                  onClick={() => selectSamplePoint(pt)}
                  disabled={isLoading}
                  title={`${pt.name} (${pt.latitude}°N, ${pt.longitude}°E): ${pt.description}`}
                  className={`text-[11px] px-2.5 py-1 rounded-md border transition-all ${
                    latitude === pt.latitude && longitude === pt.longitude
                      ? "bg-cyan-50 border-cyan-500 text-cyan-700 font-semibold"
                      : "bg-slate-50 border-slate-200 text-slate-600 hover:bg-slate-100 hover:text-slate-900"
                  }`}
                >
                  {pt.name}
                </button>
              ))}
            </div>
          </div>
        )}

        {/* Submit Button */}
        <button
          id="predict-submit-button"
          type="submit"
          disabled={isLoading}
          className={`w-full py-2.5 px-4 rounded-lg font-medium text-sm text-white shadow-sm flex items-center justify-center space-x-2 transition-all ${
            isLoading
              ? "bg-slate-400 cursor-not-allowed"
              : "bg-cyan-600 hover:bg-cyan-700 active:scale-[0.99] focus:outline-none focus:ring-2 focus:ring-cyan-500 focus:ring-offset-2"
          }`}
        >
          {isLoading ? (
            <>
              <Loader2 className="w-4 h-4 animate-spin" />
              <span>Predicting Temperature Profile...</span>
            </>
          ) : (
            <>
              <Play className="w-4 h-4 fill-current" />
              <span>Predict 15-Depth Profile</span>
            </>
          )}
        </button>
      </form>
    </div>
  );
};

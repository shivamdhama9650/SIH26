"use client";

import React from "react";
import { Info, CheckCircle2, Clock, Grid3X3, Database, ShieldAlert } from "lucide-react";
import { PredictionResponse } from "@/lib/api";

interface PredictionMetadataProps {
  prediction: PredictionResponse | null;
}

export const PredictionMetadata: React.FC<PredictionMetadataProps> = ({ prediction }) => {
  if (!prediction) return null;

  const isDemo = prediction.is_demo || prediction.mode === "mock";

  return (
    <div className="bg-white border border-slate-200 rounded-xl p-5 shadow-sm space-y-4">
      <div className="flex items-center justify-between pb-3 border-b border-slate-100">
        <div className="flex items-center space-x-2">
          <div className="p-1.5 rounded bg-slate-100 text-slate-700">
            <Info className="w-4 h-4" />
          </div>
          <h2 className="text-sm font-semibold uppercase tracking-wider text-slate-800">
            Prediction Metadata
          </h2>
        </div>
        <div className="flex items-center space-x-1.5 text-xs text-emerald-600 bg-emerald-50 px-2 py-0.5 rounded-full font-medium">
          <CheckCircle2 className="w-3.5 h-3.5" />
          <span>Inference Complete</span>
        </div>
      </div>

      {isDemo && (
        <div className="bg-amber-50 border border-amber-200 rounded-lg p-3 text-xs text-amber-800 flex items-start space-x-2">
          <ShieldAlert className="w-4 h-4 text-amber-600 shrink-0 mt-0.5" />
          <div>
            <span className="font-semibold">DEMO MODE NOTICE: </span>
            <span>{prediction.warning_notice || "Operating in mock mode pending models/cnn_best.pt checkpoint."}</span>
          </div>
        </div>
      )}

      {/* Grid of metadata attributes */}
      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3 text-xs">
        {/* Location */}
        <div className="bg-slate-50 border border-slate-200/80 rounded-lg p-2.5">
          <span className="text-slate-400 block text-[10px] uppercase font-semibold">Location</span>
          <span className="font-mono text-slate-800 font-medium">
            {prediction.location.latitude.toFixed(2)}°N, {prediction.location.longitude.toFixed(2)}°E
          </span>
        </div>

        {/* Date */}
        <div className="bg-slate-50 border border-slate-200/80 rounded-lg p-2.5">
          <span className="text-slate-400 block text-[10px] uppercase font-semibold">Obs Date</span>
          <span className="font-mono text-slate-800 font-medium">{prediction.date}</span>
        </div>

        {/* Model */}
        <div className="bg-slate-50 border border-slate-200/80 rounded-lg p-2.5">
          <span className="text-slate-400 block text-[10px] uppercase font-semibold">Inference Model</span>
          <span className="font-mono text-slate-800 font-medium">
            {prediction.metadata.model_name || "OceanXRay-CNN"}
          </span>
        </div>

        {/* Input Patch */}
        <div className="bg-slate-50 border border-slate-200/80 rounded-lg p-2.5">
          <span className="text-slate-400 block text-[10px] uppercase font-semibold">Input Type</span>
          <span className="font-mono text-slate-800 font-medium flex items-center space-x-1">
            <Grid3X3 className="w-3 h-3 text-cyan-600 inline" />
            <span>3×3 Surface Patch</span>
          </span>
        </div>

        {/* Depths Count */}
        <div className="bg-slate-50 border border-slate-200/80 rounded-lg p-2.5">
          <span className="text-slate-400 block text-[10px] uppercase font-semibold">Output Depths</span>
          <span className="font-mono text-slate-800 font-medium">
            {prediction.depths.length} Vertical Levels
          </span>
        </div>

        {/* Latency */}
        <div className="bg-slate-50 border border-slate-200/80 rounded-lg p-2.5">
          <span className="text-slate-400 block text-[10px] uppercase font-semibold">Latency</span>
          <span className="font-mono text-slate-800 font-medium flex items-center space-x-1">
            <Clock className="w-3 h-3 text-slate-400" />
            <span>{prediction.metadata.inference_time_ms ?? "--"} ms</span>
          </span>
        </div>
      </div>

      {/* Surface Input Summary */}
      {prediction.surface_input_summary && (
        <div className="border-t border-slate-100 pt-3">
          <h4 className="text-xs font-semibold text-slate-600 mb-2">
            Surface Observation Context (3×3 Center Values):
          </h4>
          <div className="flex flex-wrap gap-4 text-xs font-mono text-slate-700">
            <div className="bg-slate-50 px-2.5 py-1 rounded border border-slate-200">
              <span className="text-slate-400">SST: </span>
              <span className="font-bold text-amber-600">{prediction.surface_input_summary.center_sst_celsius} °C</span>
            </div>
            <div className="bg-slate-50 px-2.5 py-1 rounded border border-slate-200">
              <span className="text-slate-400">Salinity (SSS): </span>
              <span className="font-bold text-cyan-600">{prediction.surface_input_summary.center_sss_psu} PSU</span>
            </div>
            <div className="bg-slate-50 px-2.5 py-1 rounded border border-slate-200">
              <span className="text-slate-400">Sea Level Anomaly (SSH): </span>
              <span className="font-bold text-blue-600">{prediction.surface_input_summary.center_ssh_meters} m</span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

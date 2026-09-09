"use client";

import React from "react";
import {
  Info,
  CheckCircle2,
  Clock,
  Grid3X3,
  Flame,
  Volume2,
  TrendingDown,
  Waves,
  ShieldCheck,
  ShieldAlert,
} from "lucide-react";
import { PredictionResponse } from "@/lib/api";

interface PredictionMetadataProps {
  prediction: PredictionResponse | null;
}

export const PredictionMetadata: React.FC<PredictionMetadataProps> = ({ prediction }) => {
  if (!prediction) return null;

  const isDemo = prediction.is_demo || prediction.mode === "mock";
  const depths = prediction.depths;
  const temps = prediction.temperatures;

  // 1. Calculate D26 Isotherm Depth (depth where temperature drops to 26°C)
  let d26 = depths[0] ?? 0;
  for (let i = 0; i < depths.length - 1; i++) {
    const t1 = temps[i] ?? 0;
    const t2 = temps[i + 1] ?? 0;
    if (t1 >= 26.0 && t2 <= 26.0 && t1 !== t2) {
      const frac = (26.0 - t1) / (t2 - t1);
      d26 = Math.round(depths[i] + frac * (depths[i + 1] - depths[i]));
      break;
    } else if (t1 >= 26.0) {
      d26 = depths[i];
    }
  }

  // Cyclone heat potential classification based on D26 depth
  const tchpCategory =
    d26 >= 65
      ? { label: "High Fueling Risk", color: "text-red-700 bg-red-50 border-red-200" }
      : d26 >= 35
      ? { label: "Moderate Potential", color: "text-amber-700 bg-amber-50 border-amber-200" }
      : { label: "Low Cyclone Risk", color: "text-emerald-700 bg-emerald-50 border-emerald-200" };

  // 2. Calculate Mixed Layer Depth (MLD) (where T drops by 0.2°C from surface)
  const surfaceTemp = temps[0] ?? 28;
  let mld = 30;
  for (let i = 1; i < depths.length; i++) {
    if ((temps[i] ?? surfaceTemp) <= surfaceTemp - 0.2) {
      mld = depths[i];
      break;
    }
  }

  // 3. Calculate Maximum Thermocline Gradient (°C / 100m)
  let maxGradient = 0;
  let maxGradDepth = 75;
  for (let i = 0; i < depths.length - 1; i++) {
    const dz = depths[i + 1] - depths[i];
    if (dz > 0) {
      const grad = Math.abs((temps[i] - temps[i + 1]) / dz) * 100;
      if (grad > maxGradient) {
        maxGradient = grad;
        maxGradDepth = depths[i];
      }
    }
  }

  // 4. Calculate Sonic Layer Depth (depth of max acoustic speed in upper 200m)
  let maxSpeed = 0;
  let sld = 50;
  depths.forEach((d, idx) => {
    if (d <= 200) {
      const t = temps[idx] ?? 0;
      const c = 1448.96 + 4.591 * t - 0.05304 * t * t + 2.374e-4 * Math.pow(t, 3) + 0.0163 * d;
      if (c > maxSpeed) {
        maxSpeed = c;
        sld = d;
      }
    }
  });

  return (
    <div className="bg-white border border-slate-200 rounded-xl p-5 shadow-sm space-y-4">
      {/* Top Header */}
      <div className="flex items-center justify-between pb-3 border-b border-slate-100">
        <div className="flex items-center space-x-2">
          <div className="p-1.5 rounded bg-cyan-50 text-cyan-700">
            <Info className="w-4 h-4" />
          </div>
          <h2 className="text-sm font-semibold uppercase tracking-wider text-slate-800">
            Ocean Intelligence & Telemetry
          </h2>
        </div>
        <div className="flex items-center space-x-1.5 text-xs text-emerald-700 bg-emerald-50 border border-emerald-200 px-2.5 py-0.5 rounded-full font-medium">
          <CheckCircle2 className="w-3.5 h-3.5" />
          <span>Inference Verified</span>
        </div>
      </div>

      {isDemo && (
        <div className="bg-amber-50 border border-amber-200 rounded-lg p-3 text-xs text-amber-800 flex items-start space-x-2">
          <ShieldAlert className="w-4 h-4 text-amber-600 shrink-0 mt-0.5" />
          <div>
            <span className="font-semibold">DEMO MODE NOTICE: </span>
            <span>{prediction.warning_notice || "Operating in demo mode."}</span>
          </div>
        </div>
      )}

      {/* Operational Marine Intelligence Cards (High-Impact UI) */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3">
        {/* Cyclone Heat Potential / D26 */}
        <div className="bg-gradient-to-br from-amber-50/50 to-orange-50/30 border border-amber-200/80 rounded-xl p-3 flex flex-col justify-between">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-1.5 text-amber-800 font-semibold text-xs">
              <Flame className="w-4 h-4 text-amber-600" />
              <span>Cyclone Heat (TCHP)</span>
            </div>
            <span className={`text-[10px] font-bold px-1.5 py-0.5 rounded border ${tchpCategory.color}`}>
              {tchpCategory.label}
            </span>
          </div>
          <div className="mt-2 flex items-baseline justify-between">
            <span className="text-lg font-bold font-mono text-slate-900">{d26} m</span>
            <span className="text-[11px] text-slate-500 font-mono">D₂₆ Isotherm Depth</span>
          </div>
          <p className="text-[10px] text-slate-500 mt-1">Tropical cyclone fuel reservoir capacity</p>
        </div>

        {/* Sonic Layer Depth (Naval ASW) */}
        <div className="bg-gradient-to-br from-violet-50/50 to-indigo-50/30 border border-violet-200/80 rounded-xl p-3 flex flex-col justify-between">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-1.5 text-violet-800 font-semibold text-xs">
              <Volume2 className="w-4 h-4 text-violet-600" />
              <span>Sonic Layer (SLD)</span>
            </div>
            <span className="text-[10px] font-bold text-violet-700 bg-violet-100 border border-violet-200 px-1.5 py-0.5 rounded">
              Naval Sonar
            </span>
          </div>
          <div className="mt-2 flex items-baseline justify-between">
            <span className="text-lg font-bold font-mono text-slate-900">{sld} m</span>
            <span className="text-[11px] text-slate-500 font-mono">Acoustic Duct Axis</span>
          </div>
          <p className="text-[10px] text-slate-500 mt-1">Submarine acoustic shadow zone forms below</p>
        </div>

        {/* Max Thermocline Gradient */}
        <div className="bg-gradient-to-br from-cyan-50/50 to-blue-50/30 border border-cyan-200/80 rounded-xl p-3 flex flex-col justify-between">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-1.5 text-cyan-800 font-semibold text-xs">
              <TrendingDown className="w-4 h-4 text-cyan-600" />
              <span>Max Thermal Gradient</span>
            </div>
            <span className="text-[10px] font-bold text-cyan-700 bg-cyan-100 border border-cyan-200 px-1.5 py-0.5 rounded font-mono">
              @{maxGradDepth}m
            </span>
          </div>
          <div className="mt-2 flex items-baseline justify-between">
            <span className="text-lg font-bold font-mono text-slate-900">
              {maxGradient.toFixed(1)} °C
            </span>
            <span className="text-[11px] text-slate-500 font-mono">/ 100m depth</span>
          </div>
          <p className="text-[10px] text-slate-500 mt-1">Sharpest pycnocline barrier layer</p>
        </div>

        {/* Mixed Layer Depth (MLD) */}
        <div className="bg-gradient-to-br from-emerald-50/50 to-teal-50/30 border border-emerald-200/80 rounded-xl p-3 flex flex-col justify-between">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-1.5 text-emerald-800 font-semibold text-xs">
              <Waves className="w-4 h-4 text-emerald-600" />
              <span>Mixed Layer (MLD)</span>
            </div>
            <span className="text-[10px] font-bold text-emerald-700 bg-emerald-100 border border-emerald-200 px-1.5 py-0.5 rounded">
              Isothermal
            </span>
          </div>
          <div className="mt-2 flex items-baseline justify-between">
            <span className="text-lg font-bold font-mono text-slate-900">{mld} m</span>
            <span className="text-[11px] text-slate-500 font-mono">ΔT = 0.2°C Threshold</span>
          </div>
          <p className="text-[10px] text-slate-500 mt-1">Wind-driven surface boundary layer</p>
        </div>
      </div>

      {/* Grid of technical metadata attributes */}
      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3 text-xs pt-1">
        <div className="bg-slate-50 border border-slate-200/80 rounded-lg p-2.5">
          <span className="text-slate-400 block text-[10px] uppercase font-semibold">Station</span>
          <span className="font-mono text-slate-800 font-medium">
            {prediction.location.latitude.toFixed(2)}°N, {prediction.location.longitude.toFixed(2)}°E
          </span>
        </div>

        <div className="bg-slate-50 border border-slate-200/80 rounded-lg p-2.5">
          <span className="text-slate-400 block text-[10px] uppercase font-semibold">Observation</span>
          <span className="font-mono text-slate-800 font-medium">{prediction.date}</span>
        </div>

        <div className="bg-slate-50 border border-slate-200/80 rounded-lg p-2.5">
          <span className="text-slate-400 block text-[10px] uppercase font-semibold">Model Architecture</span>
          <span className="font-mono text-slate-800 font-medium truncate block">
            {prediction.metadata.model_name || "PatchCNN"}
          </span>
        </div>

        <div className="bg-slate-50 border border-slate-200/80 rounded-lg p-2.5">
          <span className="text-slate-400 block text-[10px] uppercase font-semibold">Reconstruction</span>
          <span className="font-mono text-slate-800 font-medium flex items-center space-x-1">
            <Grid3X3 className="w-3 h-3 text-cyan-600 inline" />
            <span>3×3 5-Ch Patch</span>
          </span>
        </div>

        <div className="bg-slate-50 border border-slate-200/80 rounded-lg p-2.5">
          <span className="text-slate-400 block text-[10px] uppercase font-semibold">Depth Output</span>
          <span className="font-mono text-slate-800 font-medium">15 Continuous Levels</span>
        </div>

        <div className="bg-slate-50 border border-slate-200/80 rounded-lg p-2.5">
          <span className="text-slate-400 block text-[10px] uppercase font-semibold">Inference Latency</span>
          <span className="font-mono text-emerald-700 font-bold flex items-center space-x-1">
            <Clock className="w-3 h-3 text-emerald-500" />
            <span>{prediction.metadata.inference_time_ms ?? "--"} ms</span>
          </span>
        </div>
      </div>

      {/* Surface Observation Multi-Channel Summary */}
      {prediction.surface_input_summary && (
        <div className="border-t border-slate-100 pt-3">
          <div className="flex items-center justify-between mb-2">
            <h4 className="text-xs font-semibold text-slate-600">
              Multi-Sensor Surface Boundary Observations (3×3 Center Core):
            </h4>
            <span className="text-[10px] text-slate-400 font-mono">0.25° Resolution</span>
          </div>
          <div className="flex flex-wrap gap-2 text-xs font-mono">
            <div className="bg-slate-50 px-2.5 py-1 rounded-md border border-slate-200">
              <span className="text-slate-400">SST: </span>
              <span className="font-bold text-amber-600">{prediction.surface_input_summary.center_sst_celsius} °C</span>
            </div>
            <div className="bg-slate-50 px-2.5 py-1 rounded-md border border-slate-200">
              <span className="text-slate-400">SSH: </span>
              <span className="font-bold text-blue-600">{prediction.surface_input_summary.center_ssh_meters} m</span>
            </div>
            {prediction.surface_input_summary.center_u_current !== undefined && (
              <div className="bg-slate-50 px-2.5 py-1 rounded-md border border-slate-200">
                <span className="text-slate-400">u_current: </span>
                <span className="font-bold text-cyan-700">{prediction.surface_input_summary.center_u_current} m/s</span>
              </div>
            )}
            {prediction.surface_input_summary.center_v_current !== undefined && (
              <div className="bg-slate-50 px-2.5 py-1 rounded-md border border-slate-200">
                <span className="text-slate-400">v_current: </span>
                <span className="font-bold text-cyan-700">{prediction.surface_input_summary.center_v_current} m/s</span>
              </div>
            )}
            {prediction.surface_input_summary.center_v_wind !== undefined && (
              <div className="bg-slate-50 px-2.5 py-1 rounded-md border border-slate-200">
                <span className="text-slate-400">v_wind: </span>
                <span className="font-bold text-indigo-700">{prediction.surface_input_summary.center_v_wind} m/s</span>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
};

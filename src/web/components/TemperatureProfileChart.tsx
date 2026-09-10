"use client";

import React, { useState } from "react";
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ReferenceLine,
  Area,
} from "recharts";
import { Thermometer, Volume2, ShieldCheck, Layers, AlertTriangle, Compass } from "lucide-react";

interface TemperatureProfileChartProps {
  depths: number[];
  temperatures: number[];
  isDemo?: boolean;
  locationName?: string;
}

// Ocean layer classification helper
function getOceanLayer(depth: number): { name: string; color: string } {
  if (depth <= 30) return { name: "Surface Mixed Layer", color: "text-amber-600" };
  if (depth <= 200) return { name: "Main Thermocline", color: "text-blue-600" };
  if (depth <= 800) return { name: "Intermediate Water", color: "text-indigo-600" };
  return { name: "Deep Ocean", color: "text-slate-600" };
}

// Mackenzie (1981) empirical equation for acoustic velocity in seawater (m/s)
function calculateSoundSpeed(T: number, S: number = 35.0, D: number): number {
  return (
    1448.96 +
    4.591 * T -
    5.304e-2 * Math.pow(T, 2) +
    2.374e-4 * Math.pow(T, 3) +
    1.340 * (S - 35.0) +
    1.63e-2 * D
  );
}

// Typical empirical standard deviation across depth tiers
function getUncertainty(depth: number): number {
  if (depth <= 30) return 0.55;
  if (depth <= 200) return 1.35; // highest across dynamic thermocline
  if (depth <= 500) return 0.85;
  return 0.35; // deep ocean stable
}

interface CustomTooltipProps {
  active?: boolean;
  payload?: any[];
  isAcousticMode?: boolean;
}

const CustomTooltip: React.FC<CustomTooltipProps> = ({ active, payload, isAcousticMode = false }) => {
  if (active && payload && payload.length) {
    const data = payload[0].payload;
    const layer = getOceanLayer(data.depth);
    return (
      <div className="bg-slate-900/95 backdrop-blur-xs text-white p-3.5 rounded-xl shadow-2xl border border-slate-700 text-xs space-y-1.5 min-w-[200px]">
        <div className="flex items-center justify-between pb-1.5 border-b border-slate-700">
          <span className="text-slate-400 font-medium">Depth</span>
          <span className="font-bold text-cyan-300 font-mono text-sm">{data.depth} m</span>
        </div>

        {isAcousticMode ? (
          <div className="flex items-center justify-between pt-0.5">
            <span className="text-slate-400">Sound Speed (SVP)</span>
            <span className="font-bold text-violet-300 font-mono text-sm">
              {data.soundSpeed.toFixed(1)} m/s
            </span>
          </div>
        ) : (
          <div className="flex items-center justify-between pt-0.5">
            <span className="text-slate-400">Temperature</span>
            <span className="font-bold text-amber-300 font-mono text-sm">
              {data.temperature.toFixed(2)} °C
            </span>
          </div>
        )}

        <div className="flex items-center justify-between text-[11px] text-slate-400 pt-0.5">
          <span>Uncertainty:</span>
          <span className="font-mono text-slate-300">±{data.unc.toFixed(2)} °C</span>
        </div>

        <div className="pt-1.5 border-t border-slate-800 text-[11px] text-slate-400 flex items-center justify-between">
          <span>Zone:</span>
          <span className="font-semibold text-slate-200">{layer.name}</span>
        </div>
      </div>
    );
  }
  return null;
};

export const TemperatureProfileChart: React.FC<TemperatureProfileChartProps> = ({
  depths,
  temperatures,
  isDemo = false,
  locationName,
}) => {
  const [viewMode, setViewMode] = useState<"temperature" | "sound">("temperature");
  const [showConfidence, setShowConfidence] = useState<boolean>(true);

  // Prepare chart data with both temperature and acoustic velocity
  const chartData = depths.map((d, i) => {
    const t = temperatures[i] ?? 0;
    const unc = getUncertainty(d);
    const speed = calculateSoundSpeed(t, 35.0, d);
    return {
      depth: d,
      temperature: t,
      soundSpeed: speed,
      tempUpper: Math.round((t + unc) * 100) / 100,
      tempLower: Math.round((t - unc) * 100) / 100,
      unc: unc,
    };
  });

  // Calculate Sonic Layer Depth (depth of max sound velocity in upper 200m)
  const upperData = chartData.filter((pt) => pt.depth <= 200);
  const sldPoint = upperData.reduce((prev, curr) =>
    curr.soundSpeed > prev.soundSpeed ? curr : prev,
    upperData[0] || { depth: 50, soundSpeed: 1540 }
  );

  const minTemp = temperatures.length > 0 ? Math.floor(Math.min(...temperatures) - 1) : 0;
  const maxTemp = temperatures.length > 0 ? Math.ceil(Math.max(...temperatures) + 1) : 32;

  const minSpeed = chartData.length > 0 ? Math.floor(Math.min(...chartData.map((d) => d.soundSpeed)) - 5) : 1480;
  const maxSpeed = chartData.length > 0 ? Math.ceil(Math.max(...chartData.map((d) => d.soundSpeed)) + 5) : 1550;

  return (
    <div className="bg-white border border-slate-200 rounded-xl p-5 shadow-sm flex flex-col h-full relative">
      {/* Header & Controls Bar */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between pb-3 border-b border-slate-100 gap-3">
        <div className="flex items-center space-x-2">
          <div className={`p-1.5 rounded-lg ${viewMode === "temperature" ? "bg-amber-50 text-amber-700" : "bg-violet-50 text-violet-700"}`}>
            {viewMode === "temperature" ? <Thermometer className="w-4 h-4" /> : <Volume2 className="w-4 h-4" />}
          </div>
          <div>
            <h2 className="text-sm font-semibold uppercase tracking-wider text-slate-800 flex items-center gap-2">
              <span>{viewMode === "temperature" ? "Vertical Temperature Profile" : "Acoustic Sound Velocity Profile (SVP)"}</span>
              {viewMode === "sound" && (
                <span className="text-[10px] font-bold text-violet-700 bg-violet-100 px-1.5 py-0.5 rounded">
                  Naval ASW
                </span>
              )}
            </h2>
            <p className="text-xs text-slate-500">
              {viewMode === "temperature"
                ? "Depth (0m to 1000m) vs. In-situ Temperature (°C)"
                : "Mackenzie (1981) Equation for Subsurface Sonar Propagation"}
            </p>
          </div>
        </div>

        {/* View Mode Toggle Buttons */}
        <div className="flex items-center space-x-2">
          <div className="bg-slate-100 p-0.5 rounded-lg flex items-center text-xs font-medium border border-slate-200">
            <button
              onClick={() => setViewMode("temperature")}
              className={`px-2.5 py-1 rounded-md flex items-center space-x-1 transition-all ${
                viewMode === "temperature"
                  ? "bg-white text-slate-900 shadow-xs font-semibold"
                  : "text-slate-600 hover:text-slate-900"
              }`}
            >
              <Thermometer className="w-3.5 h-3.5 text-amber-600" />
              <span>Thermal (°C)</span>
            </button>
            <button
              onClick={() => setViewMode("sound")}
              className={`px-2.5 py-1 rounded-md flex items-center space-x-1 transition-all ${
                viewMode === "sound"
                  ? "bg-white text-violet-700 shadow-xs font-semibold"
                  : "text-slate-600 hover:text-slate-900"
              }`}
            >
              <Volume2 className="w-3.5 h-3.5 text-violet-600" />
              <span>Acoustic (SVP)</span>
            </button>
          </div>

          <button
            onClick={() => setShowConfidence(!showConfidence)}
            title="Toggle ±1σ Empirical Validation Envelope"
            className={`px-2 py-1 rounded-lg text-xs font-medium border flex items-center space-x-1 transition-all ${
              showConfidence
                ? "bg-cyan-50 border-cyan-300 text-cyan-800"
                : "bg-slate-50 border-slate-200 text-slate-500"
            }`}
          >
            <ShieldCheck className="w-3.5 h-3.5" />
            <span className="hidden md:inline">±1σ Envelope</span>
          </button>
        </div>
      </div>

      {/* Profile Chart Container */}
      <div className="flex-1 min-h-[430px] w-full mt-3 relative">
        {chartData.length === 0 ? (
          <div className="h-full flex items-center justify-center text-slate-400 text-sm">
            Submit a prediction to visualize the 15-depth vertical profile.
          </div>
        ) : (
          <ResponsiveContainer width="100%" height="100%">
            <LineChart
              layout="vertical"
              data={chartData}
              margin={{ top: 25, right: 35, bottom: 35, left: 25 }}
            >
              <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
              
              {/* X Axis */}
              {viewMode === "temperature" ? (
                <XAxis
                  type="number"
                  dataKey="temperature"
                  domain={[minTemp, maxTemp]}
                  unit="°C"
                  stroke="#64748b"
                  tick={{ fontSize: 11 }}
                  label={{
                    value: "Predicted Temperature (°C)",
                    position: "insideBottom",
                    offset: -18,
                    fill: "#475569",
                    fontSize: 12,
                    fontWeight: 600,
                  }}
                />
              ) : (
                <XAxis
                  type="number"
                  dataKey="soundSpeed"
                  domain={[minSpeed, maxSpeed]}
                  unit="m/s"
                  stroke="#64748b"
                  tick={{ fontSize: 11 }}
                  label={{
                    value: "Acoustic Velocity c (m/s) — Mackenzie Model",
                    position: "insideBottom",
                    offset: -18,
                    fill: "#6b21a8",
                    fontSize: 12,
                    fontWeight: 600,
                  }}
                />
              )}

              {/* Y Axis: Depth (m) vertically downward with clean spacing */}
              <YAxis
                type="number"
                dataKey="depth"
                reversed={true}
                domain={[-20, 1020]}
                ticks={[0, 100, 200, 300, 500, 700, 1000]}
                unit="m"
                stroke="#64748b"
                tick={{ fontSize: 11 }}
                label={{
                  value: "Ocean Depth (m, inverted)",
                  angle: -90,
                  position: "insideLeft",
                  offset: 0,
                  fill: "#475569",
                  fontSize: 12,
                  fontWeight: 600,
                }}
              />

              {/* Oceanographic Reference Markers floated cleanly above lines */}
              <ReferenceLine
                y={50}
                stroke="#0284c7"
                strokeDasharray="4 4"
                label={({ viewBox }) => (
                  <text
                    x={viewBox.x + 15}
                    y={viewBox.y - 8}
                    fill="#0284c7"
                    fontSize={11}
                    fontWeight={600}
                    textAnchor="start"
                  >
                    Mixed Layer Depth (~50m)
                  </text>
                )}
              />
              <ReferenceLine
                y={200}
                stroke="#64748b"
                strokeDasharray="4 4"
                label={({ viewBox }) => (
                  <text
                    x={viewBox.x + viewBox.width - 15}
                    y={viewBox.y - 8}
                    fill="#475569"
                    fontSize={11}
                    fontWeight={600}
                    textAnchor="end"
                  >
                    Thermocline Base (~200m)
                  </text>
                )}
              />

              {viewMode === "sound" && (
                <ReferenceLine
                  y={sldPoint.depth}
                  stroke="#7c3aed"
                  strokeWidth={2}
                  strokeDasharray="2 2"
                  label={({ viewBox }) => (
                    <text
                      x={viewBox.x + 15}
                      y={viewBox.y - 8}
                      fill="#7c3aed"
                      fontSize={11}
                      fontWeight={700}
                      textAnchor="start"
                    >
                      Sonic Layer ({sldPoint.depth}m) — Shadow Zone Below
                    </text>
                  )}
                />
              )}

              <Tooltip content={<CustomTooltip isAcousticMode={viewMode === "sound"} />} />

              {/* Primary Line */}
              {viewMode === "temperature" ? (
                <Line
                  type="monotone"
                  dataKey="temperature"
                  stroke="#0284c7"
                  strokeWidth={3}
                  dot={{ r: 4, fill: "#0369a1", stroke: "#ffffff", strokeWidth: 2 }}
                  activeDot={{ r: 7, fill: "#0284c7", stroke: "#ffffff", strokeWidth: 2 }}
                  name="Temperature (°C)"
                />
              ) : (
                <Line
                  type="monotone"
                  dataKey="soundSpeed"
                  stroke="#7c3aed"
                  strokeWidth={3}
                  dot={{ r: 4, fill: "#6d28d9", stroke: "#ffffff", strokeWidth: 2 }}
                  activeDot={{ r: 7, fill: "#7c3aed", stroke: "#ffffff", strokeWidth: 2 }}
                  name="Sound Velocity (m/s)"
                />
              )}

              {/* Confidence interval lines if enabled */}
              {showConfidence && viewMode === "temperature" && (
                <>
                  <Line
                    type="monotone"
                    dataKey="tempUpper"
                    stroke="#94a3b8"
                    strokeWidth={1}
                    strokeDasharray="3 3"
                    dot={false}
                    name="Upper Bound (+1σ)"
                  />
                  <Line
                    type="monotone"
                    dataKey="tempLower"
                    stroke="#94a3b8"
                    strokeWidth={1}
                    strokeDasharray="3 3"
                    dot={false}
                    name="Lower Bound (-1σ)"
                  />
                </>
              )}
            </LineChart>
          </ResponsiveContainer>
        )}
      </div>

      {/* Layer legend footer */}
      <div className="pt-3 border-t border-slate-100 flex flex-wrap items-center justify-between text-[11px] text-slate-500 gap-2">
        <div className="flex items-center space-x-4">
          <div className="flex items-center space-x-1.5">
            <span className={`w-2.5 h-2.5 rounded-full inline-block ${viewMode === "temperature" ? "bg-cyan-600" : "bg-violet-600"}`} />
            <span>{viewMode === "temperature" ? "Thermal Profile (15 Depths)" : "Sound Velocity Profile (SVP)"}</span>
          </div>
          {showConfidence && viewMode === "temperature" && (
            <div className="flex items-center space-x-1.5 text-slate-400">
              <span className="w-3 h-0.5 border-b border-dashed border-slate-400 inline-block" />
              <span>±1σ Validation Margin</span>
            </div>
          )}
        </div>
        <span className="font-mono text-slate-400">Surface (0m) → Intermediate (1000m)</span>
      </div>
    </div>
  );
};

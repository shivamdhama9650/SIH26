"use client";

import React from "react";
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ReferenceLine,
} from "recharts";
import { Thermometer, Layers, AlertTriangle } from "lucide-react";

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

interface CustomTooltipProps {
  active?: boolean;
  payload?: any[];
  label?: any;
}

const CustomTooltip: React.FC<CustomTooltipProps> = ({ active, payload }) => {
  if (active && payload && payload.length) {
    const data = payload[0].payload;
    const layer = getOceanLayer(data.depth);
    return (
      <div className="bg-slate-900/95 backdrop-blur-xs text-white p-3 rounded-lg shadow-xl border border-slate-700 text-xs space-y-1">
        <div className="flex items-center justify-between space-x-3 pb-1.5 border-b border-slate-700">
          <span className="text-slate-400">Depth</span>
          <span className="font-bold text-cyan-300 font-mono text-sm">{data.depth} m</span>
        </div>
        <div className="flex items-center justify-between space-x-3 pt-1">
          <span className="text-slate-400">Temperature</span>
          <span className="font-bold text-amber-300 font-mono text-sm">{data.temperature.toFixed(2)} °C</span>
        </div>
        <div className="pt-1 text-[11px] text-slate-400 flex items-center justify-between">
          <span>Zone:</span>
          <span className="font-medium text-slate-200">{layer.name}</span>
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
  // Format data for vertical layout line chart
  const chartData = depths.map((d, i) => ({
    depth: d,
    temperature: temperatures[i] ?? 0,
  }));

  // Min and max for comfortable X domain
  const minTemp = temperatures.length > 0 ? Math.floor(Math.min(...temperatures) - 1) : 0;
  const maxTemp = temperatures.length > 0 ? Math.ceil(Math.max(...temperatures) + 1) : 32;

  return (
    <div className="bg-white border border-slate-200 rounded-xl p-5 shadow-sm flex flex-col h-full relative">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between pb-3 border-b border-slate-100 gap-2">
        <div className="flex items-center space-x-2">
          <div className="p-1.5 rounded bg-amber-50 text-amber-700">
            <Thermometer className="w-4 h-4" />
          </div>
          <div>
            <h2 className="text-sm font-semibold uppercase tracking-wider text-slate-800">
              Vertical Temperature Profile
            </h2>
            <p className="text-xs text-slate-500">
              Depth (0 m at surface down to 1500 m deep ocean) vs. Temperature (°C)
            </p>
          </div>
        </div>

        {isDemo ? (
          <div className="inline-flex items-center space-x-1 bg-amber-50 border border-amber-300 text-amber-800 px-2.5 py-1 rounded-md text-xs font-semibold">
            <AlertTriangle className="w-3.5 h-3.5 text-amber-600" />
            <span>DEMO DATA</span>
          </div>
        ) : (
          <div className="inline-flex items-center space-x-1 bg-emerald-50 border border-emerald-300 text-emerald-800 px-2.5 py-1 rounded-md text-xs font-semibold">
            <span>CNN PREDICTION</span>
          </div>
        )}
      </div>

      {/* Profile Chart Container */}
      <div className="flex-1 min-h-[420px] w-full mt-4 relative">
        {chartData.length === 0 ? (
          <div className="h-full flex items-center justify-center text-slate-400 text-sm">
            Submit a prediction to visualize the 15-depth temperature profile.
          </div>
        ) : (
          <ResponsiveContainer width="100%" height="100%">
            <LineChart
              layout="vertical"
              data={chartData}
              margin={{ top: 20, right: 30, bottom: 25, left: 20 }}
            >
              <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
              
              {/* X Axis: Temperature (°C) horizontally at bottom */}
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
                  offset: -15,
                  fill: "#475569",
                  fontSize: 12,
                  fontWeight: 600,
                }}
              />

              {/* Y Axis: Depth (m) vertically, reversed so 0m is at top */}
              <YAxis
                type="number"
                dataKey="depth"
                reversed={true}
                domain={[0, 1500]}
                ticks={[0, 10, 50, 100, 200, 400, 700, 1000, 1500]}
                unit="m"
                stroke="#64748b"
                tick={{ fontSize: 11 }}
                label={{
                  value: "Ocean Depth (m, inverted downward)",
                  angle: -90,
                  position: "insideLeft",
                  offset: 0,
                  fill: "#475569",
                  fontSize: 12,
                  fontWeight: 600,
                }}
              />

              {/* Thermocline boundary reference lines */}
              <ReferenceLine y={50} stroke="#94a3b8" strokeDasharray="4 4" label={{ value: "Mixed Layer (~50m)", position: "insideRight", fill: "#64748b", fontSize: 10 }} />
              <ReferenceLine y={200} stroke="#94a3b8" strokeDasharray="4 4" label={{ value: "Thermocline (~200m)", position: "insideRight", fill: "#64748b", fontSize: 10 }} />

              <Tooltip content={<CustomTooltip />} />

              <Line
                type="monotone"
                dataKey="temperature"
                stroke="#0284c7"
                strokeWidth={3}
                dot={{ r: 4, fill: "#0369a1", stroke: "#ffffff", strokeWidth: 2 }}
                activeDot={{ r: 7, fill: "#0284c7", stroke: "#ffffff", strokeWidth: 2 }}
              />
            </LineChart>
          </ResponsiveContainer>
        )}
      </div>

      {/* Layer legend footer */}
      <div className="pt-3 border-t border-slate-100 flex flex-wrap items-center justify-between text-[11px] text-slate-500 gap-2">
        <div className="flex items-center space-x-4">
          <div className="flex items-center space-x-1.5">
            <span className="w-2.5 h-2.5 rounded-full bg-cyan-600 inline-block" />
            <span>Profile Curve (15 Depths)</span>
          </div>
          <div className="flex items-center space-x-1.5">
            <span className="w-3 h-0.5 border-b border-dashed border-slate-400 inline-block" />
            <span>Thermocline Transition Zone</span>
          </div>
        </div>
        <span className="font-mono text-slate-400">Surface (0m) → Abyssal (1500m)</span>
      </div>
    </div>
  );
};

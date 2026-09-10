"use client";

import React from "react";
import { Table, Download, Thermometer, Layers } from "lucide-react";

interface DepthTableProps {
  depths: number[];
  temperatures: number[];
  locationStr?: string;
  dateStr?: string;
}

function getDepthZone(depth: number): { label: string; badgeClass: string } {
  if (depth <= 30) {
    return { label: "Surface Layer", badgeClass: "bg-amber-50 text-amber-700 border-amber-200" };
  }
  if (depth <= 200) {
    return { label: "Thermocline", badgeClass: "bg-blue-50 text-blue-700 border-blue-200" };
  }
  if (depth <= 700) {
    return { label: "Intermediate", badgeClass: "bg-indigo-50 text-indigo-700 border-indigo-200" };
  }
  return { label: "Deep Ocean", badgeClass: "bg-slate-100 text-slate-700 border-slate-200" };
}

export const DepthTable: React.FC<DepthTableProps> = ({
  depths,
  temperatures,
  locationStr = "Station",
  dateStr = "2025-01-15",
}) => {
  const exportCsv = () => {
    let csv = "Depth_m,Temperature_C,Temperature_F,Ocean_Zone\n";
    depths.forEach((d, idx) => {
      const t = temperatures[idx] ?? 0;
      const tf = (t * 9) / 5 + 32;
      const zone = getDepthZone(d).label;
      csv += `${d},${t.toFixed(2)},${tf.toFixed(2)},${zone}\n`;
    });

    const blob = new Blob([csv], { type: "text/csv;charset=utf-8;" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.setAttribute("download", `oceanxray_${locationStr}_${dateStr}.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  return (
    <div className="bg-white border border-slate-200 rounded-xl p-5 shadow-sm space-y-3 flex flex-col h-full">
      {/* Header */}
      <div className="flex items-center justify-between pb-3 border-b border-slate-100">
        <div className="flex items-center space-x-2">
          <div className="p-1.5 rounded bg-indigo-50 text-indigo-700">
            <Table className="w-4 h-4" />
          </div>
          <div>
            <h2 className="text-sm font-semibold uppercase tracking-wider text-slate-800">
              Depth-Wise Results
            </h2>
            <p className="text-xs text-slate-500">
              Complete {depths.length}-level temperature vertical profile
            </p>
          </div>
        </div>

        {depths.length > 0 && (
          <button
            onClick={exportCsv}
            title="Download CSV for scientific analysis"
            className="flex items-center space-x-1 px-2.5 py-1 text-xs font-medium text-slate-600 hover:text-slate-900 bg-slate-50 hover:bg-slate-100 border border-slate-200 rounded-md transition-colors"
          >
            <Download className="w-3.5 h-3.5" />
            <span>Export CSV</span>
          </button>
        )}
      </div>

      {/* Table Container */}
      <div className="flex-1 overflow-auto max-h-[460px] border border-slate-200 rounded-lg">
        <table className="min-w-full divide-y divide-slate-200 text-left text-xs">
          <thead className="bg-slate-50 text-slate-600 font-semibold sticky top-0 z-10">
            <tr>
              <th scope="col" className="px-3.5 py-2.5">Depth (m)</th>
              <th scope="col" className="px-3.5 py-2.5">Temperature (°C)</th>
              <th scope="col" className="px-3.5 py-2.5 hidden sm:table-cell">Gradient (°C/100m)</th>
              <th scope="col" className="px-3.5 py-2.5">Oceanic Zone</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-100 bg-white font-mono">
            {depths.map((depth, idx) => {
              const temp = temperatures[idx] ?? 0;
              const prevDepth = idx > 0 ? depths[idx - 1] : 0;
              const prevTemp = idx > 0 ? (temperatures[idx - 1] ?? temp) : temp;
              const dz = depth - prevDepth;
              const gradient = dz > 0 ? (((temp - prevTemp) / dz) * 100).toFixed(2) : "--";
              const zone = getDepthZone(depth);

              return (
                <tr
                  key={depth}
                  className="hover:bg-cyan-50/50 transition-colors"
                >
                  <td className="px-3.5 py-2 text-slate-900 font-semibold">
                    {depth} m
                  </td>
                  <td className="px-3.5 py-2">
                    <span className="font-bold text-slate-800">
                      {temp.toFixed(2)} °C
                    </span>
                    <span className="text-slate-400 text-[10px] ml-1.5 font-sans">
                      ({(((temp * 9) / 5) + 32).toFixed(1)}°F)
                    </span>
                  </td>
                  <td className="px-3.5 py-2 text-slate-500 hidden sm:table-cell">
                    {idx === 0 ? "--" : `${gradient} °C`}
                  </td>
                  <td className="px-3.5 py-2 font-sans">
                    <span
                      className={`inline-block px-2 py-0.5 text-[10px] font-medium rounded border ${zone.badgeClass}`}
                    >
                      {zone.label}
                    </span>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      <div className="pt-2 text-[11px] text-slate-400 flex items-center justify-between">
        <span>Depths dynamic from backend config</span>
        <span>Total Levels: {depths.length}</span>
      </div>
    </div>
  );
};

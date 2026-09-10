"use client";

import React, { useState } from "react";
import { BarChart3, TrendingDown, Layers, Award, Info, ZoomIn } from "lucide-react";

export const MetricsSection: React.FC = () => {
  const [selectedImg, setSelectedImg] = useState<{ src: string; title: string } | null>(null);

  return (
    <div className="bg-white border border-slate-200 rounded-xl p-6 shadow-sm space-y-6">
      {/* Header & Badges */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between pb-4 border-b border-slate-100 gap-3">
        <div className="space-y-1">
          <div className="flex items-center space-x-2">
            <div className="p-1.5 rounded-lg bg-cyan-50 text-cyan-700">
              <Award className="w-5 h-5" />
            </div>
            <h2 className="text-base font-bold text-slate-900 tracking-tight">
              Rigorous Scientific Benchmark Validation
            </h2>
          </div>
          <p className="text-xs text-slate-500 max-w-2xl">
            Independent evaluation on <strong>2,892,440 test profiles</strong> across the North Indian Ocean domain (5°N–30°N, 45°E–105°E) validating the 5-channel PatchCNN against standard marine baselines.
          </p>
        </div>

        <div className="flex flex-wrap gap-2">
          <span className="inline-flex items-center px-2.5 py-1 rounded-full text-xs font-semibold bg-emerald-50 text-emerald-700 border border-emerald-200 font-mono">
            CNN RMSE: 1.04°C
          </span>
          <span className="inline-flex items-center px-2.5 py-1 rounded-full text-xs font-medium bg-slate-100 text-slate-700 border border-slate-200 font-mono">
            MLP: 1.16°C
          </span>
          <span className="inline-flex items-center px-2.5 py-1 rounded-full text-xs font-medium bg-slate-100 text-slate-700 border border-slate-200 font-mono">
            Climatology: 1.23°C
          </span>
        </div>
      </div>

      {/* Metrics Summary Stats */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <div className="bg-slate-50 border border-slate-200/80 rounded-lg p-3.5">
          <div className="flex items-center justify-between">
            <span className="text-xs font-medium text-slate-500">Evaluation Split</span>
            <span className="text-[10px] uppercase font-bold text-cyan-700 bg-cyan-50 px-1.5 py-0.5 rounded">
              Held-Out
            </span>
          </div>
          <p className="text-lg font-bold text-slate-900 mt-1 font-mono">2,892,440</p>
          <p className="text-[11px] text-slate-500 mt-0.5">Test samples verified</p>
        </div>

        <div className="bg-slate-50 border border-slate-200/80 rounded-lg p-3.5">
          <div className="flex items-center justify-between">
            <span className="text-xs font-medium text-slate-500">PatchCNN RMSE</span>
            <span className="text-[10px] font-bold text-emerald-700 bg-emerald-50 px-1.5 py-0.5 rounded">
              -15.3% Err
            </span>
          </div>
          <p className="text-lg font-bold text-emerald-600 mt-1 font-mono">1.038 °C</p>
          <p className="text-[11px] text-slate-500 mt-0.5">Beats Climatology & MLP</p>
        </div>

        <div className="bg-slate-50 border border-slate-200/80 rounded-lg p-3.5">
          <div className="flex items-center justify-between">
            <span className="text-xs font-medium text-slate-500">Mean Abs. Error</span>
            <span className="text-[10px] font-bold text-emerald-700 bg-emerald-50 px-1.5 py-0.5 rounded">
              MAE
            </span>
          </div>
          <p className="text-lg font-bold text-slate-900 mt-1 font-mono">0.725 °C</p>
          <p className="text-[11px] text-slate-500 mt-0.5">Average deviation per depth</p>
        </div>

        <div className="bg-slate-50 border border-slate-200/80 rounded-lg p-3.5">
          <div className="flex items-center justify-between">
            <span className="text-xs font-medium text-slate-500">Depth Resolution</span>
            <span className="text-[10px] font-bold text-indigo-700 bg-indigo-50 px-1.5 py-0.5 rounded">
              Full Profile
            </span>
          </div>
          <p className="text-lg font-bold text-slate-900 mt-1 font-mono">15 Depths</p>
          <p className="text-[11px] text-slate-500 mt-0.5">0m to 1000m continuous</p>
        </div>
      </div>

      {/* Benchmark Visual Cards Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        {/* Plot 1: 3-Way Benchmark Comparison */}
        <div className="border border-slate-200 rounded-xl overflow-hidden bg-slate-50/50 flex flex-col hover:shadow-md transition-shadow">
          <div className="p-3 border-b border-slate-200 bg-white flex items-center justify-between">
            <div className="flex items-center space-x-1.5">
              <BarChart3 className="w-4 h-4 text-cyan-600" />
              <h3 className="text-xs font-bold text-slate-800">3-Way Benchmark</h3>
            </div>
            <span className="text-[10px] font-medium text-slate-400">RMSE Test</span>
          </div>
          <div
            className="relative cursor-pointer group bg-white overflow-hidden p-2 flex items-center justify-center min-h-[170px]"
            onClick={() => setSelectedImg({ src: "/assets/comparison_3way.png", title: "3-Way Model RMSE Comparison" })}
          >
            <img
              src="/assets/comparison_3way.png"
              alt="3-Way Benchmark Comparison"
              className="max-h-40 w-auto object-contain group-hover:scale-105 transition-transform duration-200"
            />
            <div className="absolute inset-0 bg-slate-900/10 opacity-0 group-hover:opacity-100 flex items-center justify-center transition-opacity">
              <span className="bg-white/90 text-slate-800 text-[11px] px-2 py-1 rounded shadow flex items-center gap-1 font-medium">
                <ZoomIn className="w-3.5 h-3.5" /> Enlarge
              </span>
            </div>
          </div>
          <div className="p-3 text-[11px] text-slate-600 flex-1 flex flex-col justify-between space-y-1">
            <p>
              <strong>What it shows:</strong> Compares OceanXRay CNN against historical Climatology (1.23°C) and single-pixel MLP (1.16°C).
            </p>
            <p className="text-slate-400 text-[10px]">
              Spatial 3x3 context delivers superior subsurface reconstruction.
            </p>
          </div>
        </div>

        {/* Plot 2: Training & Validation Loss Curves */}
        <div className="border border-slate-200 rounded-xl overflow-hidden bg-slate-50/50 flex flex-col hover:shadow-md transition-shadow">
          <div className="p-3 border-b border-slate-200 bg-white flex items-center justify-between">
            <div className="flex items-center space-x-1.5">
              <TrendingDown className="w-4 h-4 text-emerald-600" />
              <h3 className="text-xs font-bold text-slate-800">Training Dynamics</h3>
            </div>
            <span className="text-[10px] font-medium text-slate-400">Epoch 1–50</span>
          </div>
          <div
            className="relative cursor-pointer group bg-white overflow-hidden p-2 flex items-center justify-center min-h-[170px]"
            onClick={() => setSelectedImg({ src: "/assets/training_curve.png", title: "CNN Training & Validation Loss Curve" })}
          >
            <img
              src="/assets/training_curve.png"
              alt="CNN Training Loss Curve"
              className="max-h-40 w-auto object-contain group-hover:scale-105 transition-transform duration-200"
            />
            <div className="absolute inset-0 bg-slate-900/10 opacity-0 group-hover:opacity-100 flex items-center justify-center transition-opacity">
              <span className="bg-white/90 text-slate-800 text-[11px] px-2 py-1 rounded shadow flex items-center gap-1 font-medium">
                <ZoomIn className="w-3.5 h-3.5" /> Enlarge
              </span>
            </div>
          </div>
          <div className="p-3 text-[11px] text-slate-600 flex-1 flex flex-col justify-between space-y-1">
            <p>
              <strong>What it shows:</strong> Normalized MSE loss progression on training vs held-out validation sets.
            </p>
            <p className="text-slate-400 text-[10px]">
              Smooth convergence; early stopping at epoch 5 prevents memorization.
            </p>
          </div>
        </div>

        {/* Plot 3: Depth-Wise RMSE Profile */}
        <div className="border border-slate-200 rounded-xl overflow-hidden bg-slate-50/50 flex flex-col hover:shadow-md transition-shadow">
          <div className="p-3 border-b border-slate-200 bg-white flex items-center justify-between">
            <div className="flex items-center space-x-1.5">
              <Layers className="w-4 h-4 text-indigo-600" />
              <h3 className="text-xs font-bold text-slate-800">Depth-wise Error</h3>
            </div>
            <span className="text-[10px] font-medium text-slate-400">0–1000m</span>
          </div>
          <div
            className="relative cursor-pointer group bg-white overflow-hidden p-2 flex items-center justify-center min-h-[170px]"
            onClick={() => setSelectedImg({ src: "/assets/depthwise_rmse.png", title: "Depth-Wise RMSE Across 15 Depths" })}
          >
            <img
              src="/assets/depthwise_rmse.png"
              alt="Depth-wise RMSE Profile"
              className="max-h-40 w-auto object-contain group-hover:scale-105 transition-transform duration-200"
            />
            <div className="absolute inset-0 bg-slate-900/10 opacity-0 group-hover:opacity-100 flex items-center justify-center transition-opacity">
              <span className="bg-white/90 text-slate-800 text-[11px] px-2 py-1 rounded shadow flex items-center gap-1 font-medium">
                <ZoomIn className="w-3.5 h-3.5" /> Enlarge
              </span>
            </div>
          </div>
          <div className="p-3 text-[11px] text-slate-600 flex-1 flex flex-col justify-between space-y-1">
            <p>
              <strong>What it shows:</strong> Error distribution at each individual depth level down to 1000m.
            </p>
            <p className="text-slate-400 text-[10px]">
              Strongest accuracy gains achieved across the complex thermocline zone (50–200m).
            </p>
          </div>
        </div>

        {/* Plot 4: Latent Embedding PCA */}
        <div className="border border-slate-200 rounded-xl overflow-hidden bg-slate-50/50 flex flex-col hover:shadow-md transition-shadow">
          <div className="p-3 border-b border-slate-200 bg-white flex items-center justify-between">
            <div className="flex items-center space-x-1.5">
              <Info className="w-4 h-4 text-amber-600" />
              <h3 className="text-xs font-bold text-slate-800">Embedding PCA</h3>
            </div>
            <span className="text-[10px] font-medium text-slate-400">32-d Latent</span>
          </div>
          <div
            className="relative cursor-pointer group bg-white overflow-hidden p-2 flex items-center justify-center min-h-[170px]"
            onClick={() => setSelectedImg({ src: "/assets/embedding_pca.png", title: "Latent Embedding 2D PCA Projection" })}
          >
            <img
              src="/assets/embedding_pca.png"
              alt="Latent Embedding PCA"
              className="max-h-40 w-auto object-contain group-hover:scale-105 transition-transform duration-200"
            />
            <div className="absolute inset-0 bg-slate-900/10 opacity-0 group-hover:opacity-100 flex items-center justify-center transition-opacity">
              <span className="bg-white/90 text-slate-800 text-[11px] px-2 py-1 rounded shadow flex items-center gap-1 font-medium">
                <ZoomIn className="w-3.5 h-3.5" /> Enlarge
              </span>
            </div>
          </div>
          <div className="p-3 text-[11px] text-slate-600 flex-1 flex flex-col justify-between space-y-1">
            <p>
              <strong>What it shows:</strong> 2D PCA projection of the 32-dimensional latent feature space learned by the CNN encoder.
            </p>
            <p className="text-slate-400 text-[10px]">
              Confirms the network autonomously groups coherent water masses and regional ocean dynamics.
            </p>
          </div>
        </div>
      </div>

      {/* Modal for enlarged image preview */}
      {selectedImg && (
        <div
          className="fixed inset-0 z-50 bg-slate-900/75 backdrop-blur-sm flex items-center justify-center p-4"
          onClick={() => setSelectedImg(null)}
        >
          <div
            className="bg-white rounded-xl max-w-3xl w-full p-5 space-y-3 shadow-2xl relative"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="flex items-center justify-between pb-2 border-b border-slate-100">
              <h3 className="text-sm font-bold text-slate-800">{selectedImg.title}</h3>
              <button
                onClick={() => setSelectedImg(null)}
                className="text-slate-400 hover:text-slate-600 px-2 py-0.5 rounded text-sm font-bold"
              >
                ✕
              </button>
            </div>
            <div className="flex items-center justify-center bg-slate-50 p-3 rounded-lg">
              <img
                src={selectedImg.src}
                alt={selectedImg.title}
                className="max-h-[70vh] w-auto object-contain rounded"
              />
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

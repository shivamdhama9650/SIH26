"use client";

import React from "react";
import { Cpu, Globe, Layers, Activity, AlertTriangle } from "lucide-react";
import { HealthResponse, ConfigResponse } from "@/lib/api";

interface StatusCardsProps {
  health: HealthResponse | null;
  config: ConfigResponse | null;
  isLoading?: boolean;
}

export const StatusCards: React.FC<StatusCardsProps> = ({ health, config, isLoading }) => {
  const isModelLoaded = health?.model_loaded === true;
  const isDemo = health?.mode === "mock" || !health;

  return (
    <div className="space-y-3">
      {/* Explicit Scientific Notice Banner when in DEMO mode */}
      {isDemo && !isLoading && (
        <div className="bg-amber-500/10 border border-amber-500/30 rounded-lg p-3 flex items-start sm:items-center space-x-3 text-amber-900 dark:text-amber-300">
          <AlertTriangle className="w-5 h-5 text-amber-500 shrink-0 mt-0.5 sm:mt-0" />
          <div className="text-xs sm:text-sm">
            <span className="font-semibold uppercase tracking-wider text-amber-700 dark:text-amber-400 mr-2">
              Scientific Notice: Demo Mode Active
            </span>
            <span>
              The system is using deterministic, oceanographically realistic mock data because{" "}
              <code className="bg-amber-500/15 px-1 py-0.5 rounded font-mono text-xs">models/cnn_best.pt</code>{" "}
              is pending. Real model inference will activate seamlessly once the checkpoint is placed and{" "}
              <code className="bg-amber-500/15 px-1 py-0.5 rounded font-mono text-xs">MODEL_MODE=model</code> is set.
            </span>
          </div>
        </div>
      )}

      {/* Grid of 4 clean status cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        {/* 1. Model Status */}
        <div className="bg-white border border-slate-200 rounded-xl p-4 shadow-sm hover:border-slate-300 transition-all">
          <div className="flex items-center justify-between">
            <span className="text-xs font-medium uppercase tracking-wider text-slate-500">Model Status</span>
            <div className={`p-1.5 rounded-md ${isModelLoaded ? "bg-emerald-50 text-emerald-600" : "bg-amber-50 text-amber-600"}`}>
              <Activity className="w-4 h-4" />
            </div>
          </div>
          <div className="mt-2 flex items-center space-x-2">
            <span
              className={`inline-block w-2.5 h-2.5 rounded-full ${
                isModelLoaded ? "bg-emerald-500 ring-4 ring-emerald-100" : "bg-amber-500 ring-4 ring-amber-100"
              }`}
            />
            <h3 className="text-base font-semibold text-slate-900">
              {isModelLoaded ? "Model Loaded" : "Demo Mode"}
            </h3>
          </div>
          <p className="mt-1 text-xs text-slate-500">
            {isModelLoaded ? "Checkpoint: cnn_best.pt" : "Deterministic scientific simulation"}
          </p>
        </div>

        {/* 2. Target Domain */}
        <div className="bg-white border border-slate-200 rounded-xl p-4 shadow-sm hover:border-slate-300 transition-all">
          <div className="flex items-center justify-between">
            <span className="text-xs font-medium uppercase tracking-wider text-slate-500">Domain</span>
            <div className="p-1.5 rounded-md bg-blue-50 text-blue-600">
              <Globe className="w-4 h-4" />
            </div>
          </div>
          <h3 className="mt-2 text-base font-semibold text-slate-900">
            North Indian Ocean
          </h3>
          <p className="mt-1 text-xs text-slate-500 font-mono">
            {config ? `${config.latitude_min}°N–${config.latitude_max}°N | ${config.longitude_min}°E–${config.longitude_max}°E` : "5°N–30°N | 45°E–105°E"}
          </p>
        </div>

        {/* 3. Depth Levels */}
        <div className="bg-white border border-slate-200 rounded-xl p-4 shadow-sm hover:border-slate-300 transition-all">
          <div className="flex items-center justify-between">
            <span className="text-xs font-medium uppercase tracking-wider text-slate-500">Depth Levels</span>
            <div className="p-1.5 rounded-md bg-indigo-50 text-indigo-600">
              <Layers className="w-4 h-4" />
            </div>
          </div>
          <div className="mt-2 flex items-baseline space-x-2">
            <h3 className="text-base font-semibold text-slate-900">
              {config?.num_depths ?? 15} Standard Levels
            </h3>
          </div>
          <p className="mt-1 text-xs text-slate-500">
            Vertical range: 0 m to 1500 m
          </p>
        </div>

        {/* 4. Architecture */}
        <div className="bg-white border border-slate-200 rounded-xl p-4 shadow-sm hover:border-slate-300 transition-all">
          <div className="flex items-center justify-between">
            <span className="text-xs font-medium uppercase tracking-wider text-slate-500">Architecture</span>
            <div className="p-1.5 rounded-md bg-cyan-50 text-cyan-600">
              <Cpu className="w-4 h-4" />
            </div>
          </div>
          <h3 className="mt-2 text-base font-semibold text-slate-900">
            CNN + MLP Decoder
          </h3>
          <p className="mt-1 text-xs text-slate-500">
            3×3 Surface Patch → 15-Depth Profile
          </p>
        </div>
      </div>
    </div>
  );
};

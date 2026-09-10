"use client";

import React from "react";
import Link from "next/link";
import { DashboardHeader } from "@/components/DashboardHeader";
import {
  Waves,
  ArrowRight,
  ShieldCheck,
  Cpu,
  Layers,
  Globe,
  Database,
  CheckCircle,
  AlertTriangle,
} from "lucide-react";

export default function AboutPage() {
  return (
    <div className="min-h-screen bg-slate-50 flex flex-col">
      <DashboardHeader isBackendConnected={true} mode="mock" />

      <main className="flex-1 max-w-5xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-8">
        {/* Header Hero */}
        <div className="bg-slate-900 text-white rounded-2xl p-8 sm:p-10 shadow-lg relative overflow-hidden">
          <div className="absolute -right-10 -bottom-10 opacity-10 pointer-events-none">
            <Waves className="w-96 h-96 text-cyan-400" />
          </div>

          <div className="relative z-10 max-w-2xl space-y-4">
            <div className="inline-flex items-center space-x-2 bg-cyan-950/80 border border-cyan-800 text-cyan-300 text-xs px-3 py-1 rounded-full">
              <Cpu className="w-3.5 h-3.5" />
              <span>Deep Learning for Satellite Physical Oceanography</span>
            </div>
            <h1 className="text-3xl sm:text-4xl font-extrabold tracking-tight">
              OceanXRay
            </h1>
            <p className="text-slate-300 text-sm sm:text-base leading-relaxed">
              Inferring 15-depth vertical ocean temperature profiles across the North Indian Ocean
              from satellite-derived surface patches using deep convolutional encoders and non-linear decoders.
            </p>

            <div className="pt-2">
              <Link
                href="/dashboard"
                className="inline-flex items-center space-x-2 bg-cyan-600 hover:bg-cyan-500 text-white px-5 py-2.5 rounded-lg text-sm font-semibold transition-all shadow-md"
              >
                <span>Launch Prediction Dashboard</span>
                <ArrowRight className="w-4 h-4" />
              </Link>
            </div>
          </div>
        </div>

        {/* 4 Architecture & Research Pillars */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* Objective */}
          <div className="bg-white border border-slate-200 rounded-xl p-6 shadow-sm space-y-3">
            <div className="flex items-center space-x-2 text-cyan-700">
              <Globe className="w-5 h-5" />
              <h2 className="text-base font-bold text-slate-900">Scientific Objective</h2>
            </div>
            <p className="text-xs sm:text-sm text-slate-600 leading-relaxed">
              While satellites provide high-resolution continuous measurements of sea surface temperature (SST),
              salinity (SSS), and sea level anomaly (SSH), subsurface ocean dynamics remain largely unobserved
              in real-time. OceanXRay reconstructs the subsurface 3D thermal structure down to 1500 meters.
            </p>
          </div>

          {/* Geographical Domain */}
          <div className="bg-white border border-slate-200 rounded-xl p-6 shadow-sm space-y-3">
            <div className="flex items-center space-x-2 text-blue-700">
              <Layers className="w-5 h-5" />
              <h2 className="text-base font-bold text-slate-900">Study Region</h2>
            </div>
            <div className="text-xs sm:text-sm text-slate-600 space-y-1">
              <p className="font-semibold text-slate-800">North Indian Ocean Basin</p>
              <ul className="list-disc list-inside space-y-1 text-slate-600 text-xs">
                <li>Latitude: <strong className="text-slate-800">5°N to 30°N</strong></li>
                <li>Longitude: <strong className="text-slate-800">45°E to 105°E</strong></li>
                <li>Basins: Arabian Sea, Bay of Bengal, Equatorial Channel, and Andaman Sea</li>
              </ul>
            </div>
          </div>

          {/* Locked ML Pipeline */}
          <div className="bg-white border border-slate-200 rounded-xl p-6 shadow-sm space-y-3">
            <div className="flex items-center space-x-2 text-indigo-700">
              <Cpu className="w-5 h-5" />
              <h2 className="text-base font-bold text-slate-900">Locked ML Architecture</h2>
            </div>
            <div className="bg-slate-50 border border-slate-200 rounded-lg p-3 text-xs font-mono text-slate-700 space-y-1">
              <div className="flex items-center space-x-1 text-cyan-700 font-semibold">
                <span>3×3 Satellite Surface Patch</span>
              </div>
              <div className="text-slate-400 pl-4">&darr; 2D Convolutional Layers</div>
              <div className="text-slate-800 font-semibold pl-4">CNN Encoder</div>
              <div className="text-slate-400 pl-4">&darr; Spatial Pooling & Projection</div>
              <div className="text-slate-800 font-semibold pl-4">Latent Embedding (128-d)</div>
              <div className="text-slate-400 pl-4">&darr; Multi-Layer Perceptron</div>
              <div className="text-slate-800 font-semibold pl-4">MLP Decoder</div>
              <div className="text-slate-400 pl-4">&darr; Final Linear Projection</div>
              <div className="text-emerald-700 font-semibold">15-Depth Temperature Profile</div>
            </div>
          </div>

          {/* ARGO Scientific Integrity */}
          <div className="bg-white border border-slate-200 rounded-xl p-6 shadow-sm space-y-3">
            <div className="flex items-center space-x-2 text-emerald-700">
              <ShieldCheck className="w-5 h-5" />
              <h2 className="text-base font-bold text-slate-900">Scientific Validation Protocol</h2>
            </div>
            <p className="text-xs sm:text-sm text-slate-600 leading-relaxed">
              ARGO autonomous profiling float data is strictly post-hoc independent validation.
              In accordance with research integrity standards:
            </p>
            <div className="bg-slate-50 border border-slate-200 rounded-lg p-3 text-xs space-y-1.5 text-slate-600">
              <div className="flex items-center space-x-1.5 text-slate-700">
                <CheckCircle className="w-3.5 h-3.5 text-emerald-600" />
                <span>Never used for model training</span>
              </div>
              <div className="flex items-center space-x-1.5 text-slate-700">
                <CheckCircle className="w-3.5 h-3.5 text-emerald-600" />
                <span>Never used for data normalization</span>
              </div>
              <div className="flex items-center space-x-1.5 text-slate-700">
                <CheckCircle className="w-3.5 h-3.5 text-emerald-600" />
                <span>Never used for hyperparameter or checkpoint tuning</span>
              </div>
            </div>
          </div>
        </div>

        {/* 15 Standard Depths Reference */}
        <div className="bg-white border border-slate-200 rounded-xl p-6 shadow-sm space-y-3">
          <h2 className="text-base font-bold text-slate-900">15 Standard Oceanographic Depths</h2>
          <div className="flex flex-wrap gap-2 text-xs font-mono">
            {[0, 10, 20, 30, 50, 75, 100, 150, 200, 300, 400, 500, 700, 1000, 1500].map((depth) => (
              <span key={depth} className="bg-slate-100 border border-slate-200 px-2.5 py-1 rounded text-slate-800">
                {depth} m
              </span>
            ))}
          </div>
          <p className="text-xs text-slate-500 pt-1">
            Reconstructed depths capture the isothermal surface mixed layer (0–30m), the intense thermocline (50–200m),
            the intermediate circulation layer (300–700m), and the abyssal deep water (1000–1500m).
          </p>
        </div>
      </main>
    </div>
  );
}

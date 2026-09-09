"use client";

import React from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { Waves, Info, LayoutDashboard, ShieldCheck, AlertCircle } from "lucide-react";

interface HeaderProps {
  isBackendConnected?: boolean;
  mode?: string;
}

export const DashboardHeader: React.FC<HeaderProps> = ({
  isBackendConnected = true,
  mode = "mock",
}) => {
  const pathname = usePathname();

  return (
    <header className="bg-slate-900 border-b border-slate-800 text-white shadow-sm sticky top-0 z-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          {/* Brand */}
          <div className="flex items-center space-x-3">
            <div className="w-10 h-10 rounded-lg bg-gradient-to-tr from-cyan-600 to-blue-600 flex items-center justify-center text-white shadow-md shadow-cyan-900/30">
              <Waves className="w-6 h-6 animate-pulse" />
            </div>
            <div>
              <div className="flex items-center space-x-2">
                <span className="text-xl font-bold tracking-tight text-white">OceanXRay</span>
                <span className="text-[10px] font-semibold tracking-wider uppercase px-2 py-0.5 rounded bg-cyan-950 text-cyan-300 border border-cyan-800/60">
                  ML Inference v1.0
                </span>
              </div>
              <p className="text-xs text-slate-400 hidden sm:block">
                North Indian Ocean 15-Depth Temperature Profile Prediction
              </p>
            </div>
          </div>

          {/* Navigation & Status Badge */}
          <div className="flex items-center space-x-4">
            <nav className="flex space-x-1 sm:space-x-2">
              <Link
                href="/dashboard"
                className={`flex items-center space-x-1.5 px-3 py-1.5 rounded-md text-xs sm:text-sm font-medium transition-colors ${
                  pathname === "/dashboard" || pathname === "/"
                    ? "bg-slate-800 text-cyan-400 border border-slate-700"
                    : "text-slate-300 hover:bg-slate-800/60 hover:text-white"
                }`}
              >
                <LayoutDashboard className="w-4 h-4" />
                <span>Dashboard</span>
              </Link>

              <Link
                href="/about"
                className={`flex items-center space-x-1.5 px-3 py-1.5 rounded-md text-xs sm:text-sm font-medium transition-colors ${
                  pathname === "/about"
                    ? "bg-slate-800 text-cyan-400 border border-slate-700"
                    : "text-slate-300 hover:bg-slate-800/60 hover:text-white"
                }`}
              >
                <Info className="w-4 h-4" />
                <span>About</span>
              </Link>
            </nav>

            {/* Backend Connectivity Status Indicator */}
            <div className="hidden md:flex items-center pl-3 border-l border-slate-800">
              {isBackendConnected ? (
                <div className="flex items-center space-x-2 bg-slate-950/80 border border-slate-800 px-2.5 py-1 rounded-full text-xs">
                  <span className="relative flex h-2 w-2">
                    <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
                    <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-500"></span>
                  </span>
                  <span className="text-slate-300 text-[11px] font-mono">
                    API Connected: {mode === "model" ? (
                      <span className="text-emerald-400 font-semibold">CNN Best</span>
                    ) : (
                      <span className="text-amber-400 font-semibold">Demo Mode</span>
                    )}
                  </span>
                </div>
              ) : (
                <div className="flex items-center space-x-1.5 bg-red-950/40 border border-red-800/50 px-2.5 py-1 rounded-full text-xs text-red-400">
                  <AlertCircle className="w-3.5 h-3.5" />
                  <span className="text-[11px] font-medium">API Offline</span>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </header>
  );
};

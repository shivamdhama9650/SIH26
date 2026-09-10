"use client";

import React, { useState, useEffect, useCallback } from "react";
import { DashboardHeader } from "@/components/DashboardHeader";
import { StatusCards } from "@/components/StatusCards";
import { PredictionForm } from "@/components/PredictionForm";
import { OceanMap } from "@/components/OceanMap";
import { TemperatureProfileChart } from "@/components/TemperatureProfileChart";
import { DepthTable } from "@/components/DepthTable";
import { MetricsSection } from "@/components/MetricsSection";
import { PredictionMetadata } from "@/components/PredictionMetadata";
import { ErrorMessage } from "@/components/ErrorMessage";
import { LoadingState } from "@/components/LoadingState";
import {
  getHealth,
  getConfig,
  getSamplePoints,
  predictProfile,
  HealthResponse,
  ConfigResponse,
  SamplePoint,
  PredictionResponse,
} from "@/lib/api";

export default function DashboardPage() {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [config, setConfig] = useState<ConfigResponse | null>(null);
  const [samplePoints, setSamplePoints] = useState<SamplePoint[]>([]);

  // Input states
  const [latitude, setLatitude] = useState<number>(18.5);
  const [longitude, setLongitude] = useState<number>(65.0);
  const [date, setDate] = useState<string>("2025-01-15");

  // Output states
  const [prediction, setPrediction] = useState<PredictionResponse | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [isInitialLoading, setIsInitialLoading] = useState<boolean>(true);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [isBackendConnected, setIsBackendConnected] = useState<boolean>(true);

  // Initialize and load configuration and initial prediction
  const initializeDashboard = useCallback(async () => {
    setIsInitialLoading(true);
    setErrorMessage(null);

    try {
      const [healthData, configData, pointsData] = await Promise.all([
        getHealth().catch((err) => {
          console.warn("Backend /health unreachable:", err);
          return null;
        }),
        getConfig().catch((err) => {
          console.warn("Backend /config unreachable:", err);
          return null;
        }),
        getSamplePoints().catch(() => []),
      ]);

      if (!healthData) {
        setIsBackendConnected(false);
        setErrorMessage(
          "Unable to connect to the OceanXRay FastAPI backend (http://localhost:8000). Please start the backend service."
        );
      } else {
        setIsBackendConnected(true);
        setHealth(healthData);
        setConfig(configData);
        setSamplePoints(pointsData);

        // Fetch initial prediction
        try {
          const initialPred = await predictProfile({
            latitude: 18.5,
            longitude: 65.0,
            date: "2025-01-15",
          });
          setPrediction(initialPred);
        } catch (err: any) {
          console.error("Initial prediction error:", err);
        }
      }
    } catch (err: any) {
      setErrorMessage(err.message || "Failed to initialize dashboard.");
      setIsBackendConnected(false);
    } finally {
      setIsInitialLoading(false);
    }
  }, []);

  useEffect(() => {
    initializeDashboard();
  }, [initializeDashboard]);

  // Execute prediction
  const handlePredict = async () => {
    setIsLoading(true);
    setErrorMessage(null);

    try {
      const result = await predictProfile({
        latitude,
        longitude,
        date,
      });
      setPrediction(result);
      setIsBackendConnected(true);
    } catch (err: any) {
      setErrorMessage(
        err.message || "Unable to generate prediction. The model service is currently unavailable."
      );
    } finally {
      setIsLoading(false);
    }
  };

  const handleMapLocationSelect = (newLat: number, newLon: number) => {
    setLatitude(newLat);
    setLongitude(newLon);
  };

  return (
    <div className="min-h-screen bg-slate-50 flex flex-col selection:bg-cyan-500 selection:text-white">
      {/* Header */}
      <DashboardHeader
        isBackendConnected={isBackendConnected}
        mode={health?.mode ?? "mock"}
      />

      {/* Main Container */}
      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-6 space-y-6">
        {/* Status Cards */}
        <StatusCards
          health={health}
          config={config}
          isLoading={isInitialLoading}
        />

        {/* Global Error Banner if any */}
        <ErrorMessage
          message={errorMessage}
          onRetry={initializeDashboard}
          onDismiss={() => setErrorMessage(null)}
        />

        {/* Top Grid: Controls + Interactive Map */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-stretch">
          {/* Controls Form (5 cols on lg) */}
          <div className="lg:col-span-5 flex flex-col">
            <PredictionForm
              config={config}
              samplePoints={samplePoints}
              latitude={latitude}
              longitude={longitude}
              date={date}
              onLatitudeChange={setLatitude}
              onLongitudeChange={setLongitude}
              onDateChange={setDate}
              onSubmit={handlePredict}
              isLoading={isLoading}
            />
          </div>

          {/* Interactive North Indian Ocean Map (7 cols on lg) */}
          <div className="lg:col-span-7 flex flex-col">
            <OceanMap
              latitude={latitude}
              longitude={longitude}
              config={config}
              onLocationSelect={handleMapLocationSelect}
            />
          </div>
        </div>

        {/* Loading Overlay State during inference */}
        {isLoading && (
          <div className="py-2">
            <LoadingState />
          </div>
        )}

        {/* Bottom Grid: Visual Results (Chart + Depth Table) */}
        {prediction && (
          <div className="space-y-6">
            <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-stretch">
              {/* Profile Chart (7 cols) */}
              <div className="lg:col-span-7 flex flex-col">
                <TemperatureProfileChart
                  depths={prediction.depths}
                  temperatures={prediction.temperatures}
                  isDemo={prediction.is_demo || prediction.mode === "mock"}
                  locationName={`${prediction.location.latitude}°N, ${prediction.location.longitude}°E`}
                />
              </div>

              {/* Depth Table (5 cols) */}
              <div className="lg:col-span-5 flex flex-col">
                <DepthTable
                  depths={prediction.depths}
                  temperatures={prediction.temperatures}
                  locationStr={`${prediction.location.latitude}N_${prediction.location.longitude}E`}
                  dateStr={prediction.date}
                />
              </div>
            </div>

            {/* Prediction Metadata Card */}
            <PredictionMetadata prediction={prediction} />
            <MetricsSection />
          </div>
        )}
      </main>

      {/* Footer */}
      <footer className="bg-white border-t border-slate-200 py-4 mt-12 text-center text-xs text-slate-500">
        <div className="max-w-7xl mx-auto px-4 flex flex-col sm:flex-row items-center justify-between gap-2">
          <span>
            OceanXRay &copy; Research Project &bull; North Indian Ocean Subsurface Temperature Inference
          </span>
          <span className="text-slate-400">
            Independent ARGO Validation Pipeline Maintained
          </span>
        </div>
      </footer>
    </div>
  );
}

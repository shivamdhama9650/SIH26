export interface HealthResponse {
  status: string;
  model_loaded: boolean;
  mode: "model" | "mock" | string;
  model_path: string;
}

export interface ConfigResponse {
  latitude_min: number;
  latitude_max: number;
  longitude_min: number;
  longitude_max: number;
  num_depths: number;
  depths: number[];
  patch_size: number;
  domain_name: string;
  architecture: string;
}

export interface SamplePoint {
  name: string;
  latitude: number;
  longitude: number;
  region: string;
  description: string;
}

export interface PredictionPayload {
  latitude: number;
  longitude: number;
  date: string;
}

export interface LocationData {
  latitude: number;
  longitude: number;
}

export interface PredictionResponse {
  success: boolean;
  mode: "model" | "mock" | string;
  is_demo: boolean;
  location: LocationData;
  date: string;
  depths: number[];
  temperatures: number[];
  surface_input_summary?: {
    patch_size: string;
    center_sst_celsius: number;
    center_sss_psu: number;
    center_ssh_meters: number;
    channels: string[];
  };
  metadata: {
    model_name: string;
    checkpoint?: string;
    input_patch: string;
    inference_time_ms: number;
    scientific_notice: string;
    [key: string]: any;
  };
  warning_notice?: string | null;
}

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

async function fetchJson<T>(endpoint: string, options?: RequestInit): Promise<T> {
  const url = `${API_BASE}${endpoint}`;
  try {
    const res = await fetch(url, {
      ...options,
      headers: {
        "Content-Type": "application/json",
        ...(options?.headers || {}),
      },
    });

    if (!res.ok) {
      let errorDetail = `Request failed with status ${res.status}`;
      try {
        const errorData = await res.json();
        if (errorData.detail) {
          errorDetail = errorData.detail;
        }
      } catch {
        // Fallback to text status
      }
      throw new Error(errorDetail);
    }

    return (await res.json()) as T;
  } catch (err: any) {
    if (err.message && err.message.includes("Failed to fetch")) {
      throw new Error("Unable to reach backend service. Please verify that the FastAPI backend is running on port 8000.");
    }
    throw err;
  }
}

export async function getHealth(): Promise<HealthResponse> {
  return fetchJson<HealthResponse>("/health");
}

export async function getConfig(): Promise<ConfigResponse> {
  return fetchJson<ConfigResponse>("/config");
}

export async function getSamplePoints(): Promise<SamplePoint[]> {
  return fetchJson<SamplePoint[]>("/sample-points");
}

export async function predictProfile(payload: PredictionPayload): Promise<PredictionResponse> {
  return fetchJson<PredictionResponse>("/predict", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

# OceanXRay 🌊

**OceanXRay** is a physical oceanography deep learning platform that predicts 15-depth vertical ocean temperature profiles across the North Indian Ocean from satellite-derived surface observations.

The platform provides a minimal, high-performance **FastAPI** inference backend paired with a scientific **Next.js & TypeScript** monitoring dashboard.

---

## 🔬 Scientific Context & Domain

- **Target Domain**: North Indian Ocean
  - Latitude: **5.0°N to 30.0°N**
  - Longitude: **45.0°E to 105.0°E**
  - Basins: Arabian Sea, Bay of Bengal, Equatorial Channel, and Andaman Sea
- **Locked ML Architecture**:
  ```
  3×3 Satellite Surface Patch (SST, SSS, SSH, Coordinate Proxy)
                   ↓
              CNN Encoder
                   ↓
          Latent Embedding (128-d)
                   ↓
              MLP Decoder
                   ↓
     15-Depth Temperature Profile (°C)
  ```
- **15 Standard Oceanographic Depths**:
  `[0m, 10m, 20m, 30m, 50m, 75m, 100m, 150m, 200m, 300m, 400m, 500m, 700m, 1000m, 1500m]`
- **Scientific Integrity Protocol**:
  ARGO autonomous profiling float data is strictly post-hoc independent validation. In accordance with physical oceanography research standards, ARGO data is **never** utilized during training, normalization, hyperparameter tuning, or checkpoint selection.

---

## 📁 Repository Structure

```
SIH26/
├── backend/
│   ├── config.py           # Domain boundaries, depth levels & settings
│   ├── main.py             # FastAPI app, CORS, routes & validation
│   ├── model_service.py    # PyTorch OceanCNNEncoderDecoder & ModelService singleton
│   ├── preprocessing.py    # 3×3 surface patch builder & normalizer
│   ├── schemas.py          # Pydantic request/response schemas
│   ├── requirements.txt    # Python dependencies
│   ├── .env                # Backend environment configuration
│   └── tests/
│       └── test_api.py     # Automated pytest test suite
│
├── frontend/
│   ├── app/
│   │   ├── dashboard/      # Primary interactive prediction dashboard
│   │   ├── about/          # Scientific overview and architecture docs
│   │   ├── globals.css     # Oceanographic CSS theme & Leaflet styles
│   │   └── layout.tsx      # App wrapper with scientific metadata
│   ├── components/
│   │   ├── DashboardHeader.tsx       # Navigation & backend beacon
│   │   ├── StatusCards.tsx           # Domain & model status cards
│   │   ├── PredictionForm.tsx        # Coordinate inputs & preset stations
│   │   ├── OceanMap.tsx              # Dynamic Leaflet domain map
│   │   ├── OceanMapInner.tsx         # Leaflet container & clickable marker
│   │   ├── TemperatureProfileChart.tsx # Inverted vertical depth chart
│   │   ├── DepthTable.tsx            # 15-depth results table with CSV export
│   │   ├── PredictionMetadata.tsx    # Latency, patch summary & metadata
│   │   ├── ErrorMessage.tsx          # User-friendly error banners
│   │   └── LoadingState.tsx          # Inference loading animation
│   ├── lib/
│   │   └── api.ts          # Centralized typed HTTP client
│   ├── .env.local          # Frontend API endpoint config
│   └── package.json
│
├── models/
│   ├── .gitkeep            # Directory for trained weights
│   └── cnn_best.pt         # Placed here when model training is complete
│
└── README.md
```

---

## 🚀 Quick Start

### 1. Backend Setup (FastAPI)

Ensure Python 3.10+ is installed.

```bash
cd backend

# Install dependencies (fastapi, uvicorn, torch, numpy, xarray, pydantic)
pip install -r requirements.txt

# Run automated tests
python -m pytest tests/test_api.py -v

# Start the FastAPI server with hot reload
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at:
- Documentation: [http://localhost:8000/docs](http://localhost:8000/docs)
- Health Check: [http://localhost:8000/health](http://localhost:8000/health)
- Configuration: [http://localhost:8000/config](http://localhost:8000/config)

### 2. Frontend Setup (Next.js)

Ensure Node.js 18+ is installed.

```bash
cd frontend

# Install packages
npm install

# Start the Next.js development server
npm run dev -- -p 3001
# Or run with npx
npx next dev -p 3001
```

Access the dashboard at:
- **Dashboard**: [http://localhost:3001/dashboard](http://localhost:3001/dashboard)
- **Scientific Overview**: [http://localhost:3001/about](http://localhost:3001/about)

---

## ⚙️ Switching from Mock Mode to Real Model

Because model checkpoints may be in training or hosted on remote compute clusters, OceanXRay comes configured with a deterministic oceanographic mock simulation by default.

### Mode 1: Deterministic Mock Mode (`MODEL_MODE=mock`)
- When `MODEL_MODE=mock` is set in `backend/.env`, the system executes deterministic vertical thermal equations modeling the mixed layer, thermocline, and abyssal water based on latitude, longitude, and seasonal phase.
- Clear **DEMO DATA** badges are visibly rendered across the UI and in API responses to prevent confusing demo values with real predictions.

### Mode 2: Real PyTorch CNN Model (`MODEL_MODE=model`)
To switch to real inference once your model file is ready:

1. Place your trained PyTorch checkpoint in the `models/` directory:
   ```bash
   cp /path/to/trained_weights.pt models/cnn_best.pt
   ```
2. Update `backend/.env`:
   ```env
   MODEL_MODE=model
   MODEL_PATH=models/cnn_best.pt
   FRONTEND_ORIGIN=http://localhost:3001
   ```
3. Restart the FastAPI backend.
4. The `/health` endpoint will immediately report:
   ```json
   {
     "status": "ok",
     "model_loaded": true,
     "mode": "model",
     "model_path": "models/cnn_best.pt"
   }
   ```
5. The dashboard UI will automatically switch its status badge from **Demo Mode** to **Model Loaded (CNN Best)** with zero changes needed in the frontend code.

---

## 📡 API Reference

### `GET /health`
Returns service availability, model loading state, and active mode.

**Response:**
```json
{
  "status": "ok",
  "model_loaded": false,
  "mode": "mock",
  "model_path": "models/cnn_best.pt"
}
```

### `GET /config`
Returns North Indian Ocean boundaries and depth levels.

**Response:**
```json
{
  "latitude_min": 5.0,
  "latitude_max": 30.0,
  "longitude_min": 45.0,
  "longitude_max": 105.0,
  "num_depths": 15,
  "depths": [0.0, 10.0, 20.0, 30.0, 50.0, 75.0, 100.0, 150.0, 200.0, 300.0, 400.0, 500.0, 700.0, 1000.0, 1500.0],
  "patch_size": 3,
  "domain_name": "North Indian Ocean (5°N–30°N, 45°E–105°E)",
  "architecture": "3×3 CNN Encoder + Latent Embedding + MLP Decoder"
}
```

### `POST /predict`
Performs vertical profile prediction for a given latitude, longitude, and date.

**Request:**
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "latitude": 18.50,
       "longitude": 65.00,
       "date": "2025-01-15"
     }'
```

**Response:**
```json
{
  "success": true,
  "mode": "mock",
  "is_demo": true,
  "location": {
    "latitude": 18.5,
    "longitude": 65.0
  },
  "date": "2025-01-15",
  "depths": [0.0, 10.0, 20.0, 30.0, 50.0, 75.0, 100.0, 150.0, 200.0, 300.0, 400.0, 500.0, 700.0, 1000.0, 1500.0],
  "temperatures": [25.55, 25.47, 25.39, 25.31, 22.84, 19.86, 17.26, 13.56, 11.08, 8.22, 6.78, 5.92, 5.16, 4.81, 4.69],
  "surface_input_summary": {
    "patch_size": "3x3",
    "center_sst_celsius": 25.55,
    "center_sss_psu": 35.6,
    "center_ssh_meters": -0.016,
    "channels": ["SST (°C)", "SSS (PSU)", "SSH (m)", "Normalized Latitude"]
  },
  "metadata": {
    "model_name": "OceanXRay-Mock-DemoEngine",
    "inference_time_ms": 0.72,
    "scientific_notice": "DEMO DATA: Operating in mock mode pending final models/cnn_best.pt checkpoint."
  },
  "warning_notice": "DEMO DATA: Model checkpoint 'cnn_best.pt' is not currently active. Displayed values are realistic deterministic mock estimates."
}
```

---

## 🛠️ Validation & Error Handling

- **Coordinate Bounds Checking**:
  - Rejects latitude values outside `5.0°N – 30.0°N` with `400 Bad Request`.
  - Rejects longitude values outside `45.0°E – 105.0°E` with `400 Bad Request`.
  - Rejects invalid date strings with `400 Bad Request`.
- **Fault-Tolerant Frontend**:
  - Does not crash if the backend is temporarily offline; renders a clean status banner with a retry button.
  - Re-tries connection automatically when user selects a preset or triggers an action.

---

## 🧪 Testing

Run backend tests:
```bash
python -m pytest backend/tests/test_api.py -v
```

Run frontend type check & production build:
```bash
cd frontend
npm run build
```

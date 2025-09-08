# backend/api/api_server.py
"""
Uproszczony serwer API z obsługą warstw danych
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, Dict, List
from datetime import datetime
import numpy as np
import rasterio
import logging
import json
import hashlib
from pathlib import Path

# Import modułów
import sys
sys.path.append('/content/drive/MyDrive/microclimate/backend')
from core.wind_solver import WindFieldSolver
from core.thermal_model import UrbanThermalModel
from core.data_processor import UrbanDataProcessor

app = FastAPI(title="Urban Microclimate API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global objects
wind_solver = None
thermal_model = None
data_processor = None
simulation_cache = {}

# === Models ===
class SimulationRequest(BaseModel):
    timestamp: datetime = Field(default_factory=datetime.now)
    bounds: Dict[str, float] = Field(default={"lat_min": 54.1, "lat_max": 54.2, "lon_min": 19.3, "lon_max": 19.5})
    resolution: float = Field(default=5.0)

# === Startup ===
@app.on_event("startup")
async def startup():
    global wind_solver, thermal_model, data_processor
    
    # Initialize data processor
    data_processor = UrbanDataProcessor(
        data_dir="/content/drive/MyDrive/microclimate/data/processed",
        resolution=5.0
    )
    data_processor.load_all_layers()
    
    # Initialize solvers
    shape = data_processor.shape or (200, 200)
    wind_solver = WindFieldSolver((shape[0], shape[1], 50), 5.0)
    thermal_model = UrbanThermalModel(54.156, 19.404)
    
    logging.info("System initialized")

# === Endpoints ===

@app.get("/")
async def root():
    return {
        "name": "Urban Microclimate API",
        "endpoints": [
            "/api/status",
            "/api/layer/{name}",
            "/api/simulate",
            "/api/weather"
        ]
    }

@app.get("/api/status")
async def status():
    return {
        "status": "online",
        "data_loaded": data_processor is not None,
        "cache_size": len(simulation_cache)
    }

@app.get("/api/layer/{layer_name}")
async def get_layer(layer_name: str):
    """Zwraca warstwę danych do wyświetlenia na mapie"""
    
    if not data_processor:
        raise HTTPException(404, "Data not loaded")
    
    if layer_name not in data_processor.data_layers:
        # Spróbuj załadować z pliku
        file_path = Path(f"/content/drive/MyDrive/microclimate/data/processed/{layer_name}.tif")
        if not file_path.exists():
            raise HTTPException(404, f"Layer {layer_name} not found")
        
        with rasterio.open(file_path) as src:
            data = src.read(1)
            bounds = src.bounds
    else:
        data = data_processor.data_layers[layer_name]
        bounds = data_processor.bounds
    
    # Przygotuj dane do przesłania
    # Downsample dla wydajności
    step = 2
    data_small = data[::step, ::step]
    
    return {
        "name": layer_name,
        "values": data_small.flatten().tolist(),
        "width": data_small.shape[1],
        "height": data_small.shape[0],
        "bounds": list(bounds) if bounds else [19.3, 54.1, 19.5, 54.2],
        "min": float(np.min(data)),
        "max": float(np.max(data)),
        "mean": float(np.mean(data))
    }

@app.get("/api/weather")
async def weather():
    """Pobiera dane pogodowe (mock lub Open-Meteo)"""
    
    # Przykładowe dane testowe
    return {
        "timestamp": datetime.now().isoformat(),
        "temperature": 20.5,
        "humidity": 65,
        "pressure": 1013,
        "wind_speed": 5.2,
        "wind_direction": 270,
        "cloud_cover": 0.3
    }

@app.post("/api/simulate")
async def simulate(request: SimulationRequest):
    """Uruchamia symulację"""
    
    # Cache key
    key = hashlib.md5(json.dumps(request.dict(), default=str).encode()).hexdigest()[:8]
    
    if key in simulation_cache:
        return simulation_cache[key]
    
    # Get weather
    weather_data = await weather()
    
    # Prepare geometry
    geometry = data_processor.create_urban_geometry()
    
    # Setup wind solver obstacles
    wind_solver.obstacles = np.zeros((wind_solver.nx, wind_solver.ny, wind_solver.nz))
    for i in range(min(wind_solver.nx, geometry['building_height'].shape[0])):
        for j in range(min(wind_solver.ny, geometry['building_height'].shape[1])):
            height = int(geometry['building_height'][i, j] / wind_solver.dx)
            if height > 0:
                height = min(height, wind_solver.nz - 1)
                wind_solver.obstacles[i, j, :height] = 1.0
    
    # Wind simulation
    u_in = weather_data['wind_speed'] * np.cos(np.radians(weather_data['wind_direction']))
    v_in = weather_data['wind_speed'] * np.sin(np.radians(weather_data['wind_direction']))
    
    phi = wind_solver.solve_potential_flow(u_in, v_in, max_iter=200)
    velocity = wind_solver.compute_velocity_field(phi)
    
    # Thermal simulation
    thermal_model.albedo_map = 0.2 * np.ones(geometry['surface_elevation'].shape)
    radiation = thermal_model.compute_solar_radiation(request.timestamp, weather_data['cloud_cover'])
    
    wind_surface = velocity['speed'][:, :, 2] if velocity['speed'].shape[2] > 2 else velocity['speed'][:, :, 0]
    T_surface = thermal_model.compute_surface_temperature(
        radiation,
        weather_data['temperature'],
        wind_surface[:geometry['surface_elevation'].shape[0], :geometry['surface_elevation'].shape[1]]
    )
    
    uhi = thermal_model.compute_heat_island_intensity(T_surface)
    
    # Prepare results (compressed)
    result = {
        "id": key,
        "timestamp": request.timestamp.isoformat(),
        "weather": weather_data,
        "wind_field": {
            "u_10m": velocity['u'][:, :, 2].astype(np.float16).tolist(),
            "v_10m": velocity['v'][:, :, 2].astype(np.float16).tolist(),
            "speed_10m": velocity['speed'][:, :, 2].astype(np.float16).tolist()
        },
        "thermal_field": {
            "surface_temperature": T_surface.astype(np.float16).tolist(),
            "uhi_intensity": float(uhi)
        },
        "metadata": {
            "resolution_m": request.resolution,
            "computation_time_s": 0.0
        }
    }
    
    # Cache
    simulation_cache[key] = result
    
    return result

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

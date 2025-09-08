# backend/api/api_server.py
"""
Główny serwer API dla symulacji mikroklimatu
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, Dict, List, Tuple
from datetime import datetime, timedelta
import numpy as np
import asyncio
import logging
import json
import hashlib
import time
import os
from pathlib import Path

# Import modułów systemu
import sys
sys.path.append('/content/drive/MyDrive/microclimate/backend')

from core.wind_solver import WindFieldSolver
from core.thermal_model import UrbanThermalModel
from core.data_processor import UrbanDataProcessor

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Inicjalizacja FastAPI
app = FastAPI(
    title="Urban Microclimate API",
    description="System symulacji mikroklimatu miejskiego",
    version="1.0.0"
)

# CORS dla GitHub Pages
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cache dla wyników
simulation_cache = {}
cache_ttl = 7200  # 2 godziny

# Globalne obiekty symulacji (inicjalizowane raz)
wind_solver = None
thermal_model = None
data_processor = None

# ============= Modele danych =============

class SimulationRequest(BaseModel):
    """Request symulacji"""
    timestamp: datetime = Field(default_factory=datetime.now)
    bounds: Dict[str, float] = Field(
        default={"lat_min": 54.1, "lat_max": 54.2, "lon_min": 19.3, "lon_max": 19.5}
    )
    resolution: float = Field(default=5.0, ge=1.0, le=20.0)
    weather_override: Optional[Dict] = None

class WeatherData(BaseModel):
    """Dane pogodowe"""
    timestamp: datetime
    temperature: float  # °C
    humidity: float  # %
    pressure: float  # hPa
    wind_speed: float  # m/s
    wind_direction: float  # degrees
    cloud_cover: float  # 0-1
    solar_radiation: Optional[float] = None  # W/m²

class SimulationResult(BaseModel):
    """Wynik symulacji"""
    id: str
    timestamp: datetime
    weather: Dict
    wind_field: Dict
    thermal_field: Dict
    metadata: Dict

# ============= Funkcje pomocnicze =============

def get_cache_key(params: dict) -> str:
    """Generuj klucz cache"""
    param_str = json.dumps(params, sort_keys=True, default=str)
    return hashlib.md5(param_str.encode()).hexdigest()

async def fetch_weather_openweather(lat: float, lon: float) -> WeatherData:
    """Pobierz dane z OpenWeatherMap"""
    import httpx
    
    api_key = os.environ.get("OPENWEATHER_API_KEY", "")
    if not api_key:
        logger.warning("Brak klucza OpenWeatherMap, używam danych testowych")
        return WeatherData(
            timestamp=datetime.now(),
            temperature=20.0,
            humidity=65.0,
            pressure=1013.0,
            wind_speed=5.0,
            wind_direction=270.0,
            cloud_cover=0.3
        )
    
    url = f"https://api.openweathermap.org/data/2.5/weather"
    params = {"lat": lat, "lon": lon, "appid": api_key, "units": "metric"}
    
    async with httpx.AsyncClient() as client:
        response = await client.get(url, params=params)
        
    if response.status_code == 200:
        data = response.json()
        return WeatherData(
            timestamp=datetime.now(),
            temperature=data["main"]["temp"],
            humidity=data["main"]["humidity"],
            pressure=data["main"]["pressure"],
            wind_speed=data["wind"]["speed"],
            wind_direction=data["wind"].get("deg", 0),
            cloud_cover=data["clouds"]["all"] / 100.0
        )
    else:
        raise HTTPException(status_code=500, detail="Błąd pobierania danych pogodowych")

def compress_array(arr: np.ndarray, decimals: int = 1) -> List:
    """Kompresuj array NumPy do JSON"""
    # Zaokrąglij i konwertuj do listy
    compressed = np.around(arr, decimals=decimals).astype(np.float16)
    return compressed.tolist()

# ============= Endpoints API =============

@app.on_event("startup")
async def startup_event():
    """Inicjalizacja przy starcie"""
    global wind_solver, thermal_model, data_processor
    
    logger.info("Inicjalizacja systemu...")
    
    # Ładowanie danych
    data_processor = UrbanDataProcessor(
        data_dir="/content/drive/MyDrive/microclimate/data/processed",
        resolution=5.0
    )
    data_processor.load_all_layers()
    
    # Inicjalizacja solverów
    domain_shape = data_processor.shape
    wind_solver = WindFieldSolver(
        domain_size=(domain_shape[0], domain_shape[1], 50),  # 50 warstw pionowych
        resolution=5.0
    )
    
    thermal_model = UrbanThermalModel(lat=54.156, lon=19.404)
    
    logger.info("System zainicjalizowany")

@app.get("/")
async def root():
    """Strona główna API"""
    return {
        "name": "Urban Microclimate API",
        "status": "operational",
        "endpoints": [
            "/docs - Dokumentacja API",
            "/api/status - Status systemu",
            "/api/simulate - Uruchom symulację",
            "/api/results/{id} - Pobierz wyniki",
            "/api/weather - Aktualne dane pogodowe"
        ]
    }

@app.get("/api/status")
async def get_status():
    """Status systemu"""
    return {
        "status": "operational" if wind_solver is not None else "initializing",
        "cache_size": len(simulation_cache),
        "cache_ttl_seconds": cache_ttl,
        "server_time": datetime.now().isoformat(),
        "data_loaded": data_processor is not None,
        "solver_ready": wind_solver is not None
    }

@app.get("/api/weather")
async def get_weather():
    """Pobierz aktualne dane pogodowe"""
    weather = await fetch_weather_openweather(54.156, 19.404)
    return weather.dict()

@app.post("/api/simulate")
async def run_simulation(
    request: SimulationRequest,
    background_tasks: BackgroundTasks
):
    """Uruchom symulację mikroklimatu"""
    
    # Sprawdź cache
    cache_key = get_cache_key(request.dict())
    
    if cache_key in simulation_cache:
        cached_result = simulation_cache[cache_key]
        age = datetime.now() - cached_result["computed_at"]
        
        if age < timedelta(seconds=cache_ttl):
            logger.info(f"Cache hit dla {cache_key}")
            return cached_result["data"]
    
    logger.info(f"Rozpoczynanie symulacji {cache_key}")
    start_time = time.time()
    
    try:
        # Pobierz dane pogodowe
        if request.weather_override:
            weather = WeatherData(**request.weather_override)
        else:
            weather = await fetch_weather_openweather(
                (request.bounds["lat_min"] + request.bounds["lat_max"]) / 2,
                (request.bounds["lon_min"] + request.bounds["lon_max"]) / 2
            )
        
        # Przygotuj geometrię
        geometry = data_processor.create_urban_geometry()
        
        # Załaduj do solvera
        wind_solver.obstacles = np.zeros(wind_solver.nx * wind_solver.ny * wind_solver.nz)
        
        # Konwersja geometrii do siatki 3D
        for i in range(wind_solver.nx):
            for j in range(wind_solver.ny):
                if i < geometry['total_height'].shape[0] and j < geometry['total_height'].shape[1]:
                    height_idx = int(geometry['total_height'][i, j] / wind_solver.dx)
                    if height_idx > 0:
                        height_idx = min(height_idx, wind_solver.nz - 1)
                        wind_solver.obstacles[
                            i * wind_solver.ny * wind_solver.nz + 
                            j * wind_solver.nz : 
                            i * wind_solver.ny * wind_solver.nz + 
                            j * wind_solver.nz + height_idx
                        ] = 1.0
        
        wind_solver.obstacles = wind_solver.obstacles.reshape(
            (wind_solver.nx, wind_solver.ny, wind_solver.nz)
        )
        wind_solver.roughness = geometry['roughness_length']
        
        # Symulacja przepływu
        logger.info("Obliczanie pola wiatru...")
        u_inlet = weather.wind_speed * np.cos(np.radians(weather.wind_direction))
        v_inlet = weather.wind_speed * np.sin(np.radians(weather.wind_direction))
        
        phi = wind_solver.solve_potential_flow(u_inlet, v_inlet, max_iter=500)
        velocity_field = wind_solver.compute_velocity_field(phi)
        turbulence = wind_solver.compute_turbulence_intensity(velocity_field)
        
        # Model termiczny
        logger.info("Obliczanie pola termicznego...")
        thermal_model.albedo_map = 0.2 * np.ones(geometry['surface_elevation'].shape)  # Domyślne albedo
        
        radiation = thermal_model.compute_solar_radiation(request.timestamp, weather.cloud_cover)
        
        # Temperatura powierzchni
        wind_at_surface = velocity_field['speed'][:, :, 2]  # Warstwa 10m
        T_surface = thermal_model.compute_surface_temperature(
            radiation, 
            weather.temperature,
            wind_at_surface
        )
        
        # Urban Heat Island
        uhi_intensity = thermal_model.compute_heat_island_intensity(T_surface)
        
        # Przygotuj wyniki
        result_data = {
            "id": cache_key,
            "timestamp": request.timestamp.isoformat(),
            "weather": weather.dict(),
            "wind_field": {
                "u_10m": compress_array(velocity_field['u'][:, :, 2]),
                "v_10m": compress_array(velocity_field['v'][:, :, 2]),
                "speed_10m": compress_array(velocity_field['speed'][:, :, 2]),
                "turbulence_10m": compress_array(turbulence[:, :, 2])
            },
            "thermal_field": {
                "surface_temperature": compress_array(T_surface),
                "radiation": compress_array(radiation),
                "uhi_intensity": round(uhi_intensity, 2)
            },
            "metadata": {
                "resolution_m": request.resolution,
                "domain_size": list(geometry['surface_elevation'].shape),
                "computation_time_s": round(time.time() - start_time, 2),
                "solver_iterations": 500,
                "cache_key": cache_key
            }
        }
        
        # Zapisz do cache
        simulation_cache[cache_key] = {
            "data": result_data,
            "computed_at": datetime.now()
        }
        
        # Czyść stary cache w tle
        background_tasks.add_task(clean_old_cache)
        
        logger.info(f"Symulacja zakończona w {result_data['metadata']['computation_time_s']}s")
        
        return result_data
        
    except Exception as e:
        logger.error(f"Błąd symulacji: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Błąd symulacji: {str(e)}")

@app.get("/api/results/{simulation_id}")
async def get_results(simulation_id: str):
    """Pobierz wyniki symulacji po ID"""
    
    if simulation_id in simulation_cache:
        return simulation_cache[simulation_id]["data"]
    else:
        raise HTTPException(status_code=404, detail="Wyniki nie znalezione")

@app.delete("/api/cache")
async def clear_cache():
    """Wyczyść cache symulacji"""
    global simulation_cache
    old_size = len(simulation_cache)
    simulation_cache = {}
    
    return {
        "message": "Cache wyczyszczony",
        "cleared_items": old_size
    }

# ============= Zadania w tle =============

async def clean_old_cache():
    """Usuń stare wpisy z cache"""
    current_time = datetime.now()
    to_remove = []
    
    for key, value in simulation_cache.items():
        age = current_time - value["computed_at"]
        if age > timedelta(seconds=cache_ttl):
            to_remove.append(key)
    
    for key in to_remove:
        del simulation_cache[key]
    
    if to_remove:
        logger.info(f"Usunięto {len(to_remove)} starych wpisów z cache")

async def scheduled_simulation():
    """Automatyczna symulacja co 2h"""
    while True:
        await asyncio.sleep(7200)  # 2 godziny
        
        try:
            logger.info("Uruchamianie zaplanowanej symulacji...")
            
            request = SimulationRequest()
            await run_simulation(request, BackgroundTasks())
            
            logger.info("Zaplanowana symulacja zakończona")
        except Exception as e:
            logger.error(f"Błąd zaplanowanej symulacji: {e}")

# ============= Health checks =============

@app.get("/health")
async def health_check():
    """Endpoint dla health checków"""
    
    checks = {
        "api": True,
        "data": data_processor is not None,
        "solver": wind_solver is not None,
        "cache": len(simulation_cache) < 100  # Max 100 wpisów
    }
    
    if all(checks.values()):
        return {"status": "healthy", "checks": checks}
    else:
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "checks": checks}
        )

# Uruchomienie
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

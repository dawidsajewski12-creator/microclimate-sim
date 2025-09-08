# thermal_model.py
import numpy as np
from numba import jit
from datetime import datetime
import pvlib

class UrbanThermalModel:
    """Model bilansu cieplnego miasta z uwzględnieniem promieniowania i antropogenicznych źródeł ciepła"""
    
    def __init__(self, lat: float, lon: float):
        self.lat = lat
        self.lon = lon
        self.location = pvlib.location.Location(lat, lon, tz='UTC')
        self.albedo_map = None
        self.heat_capacity_map = None
        
    def load_surface_properties(self, landcover_path: str, resolution: Tuple[int, int]):
        """Parametryzacja właściwości termicznych powierzchni"""
        # Mapowanie klas pokrycia na właściwości fizyczne
        landcover_properties = {
            1: {'albedo': 0.15, 'heat_cap': 2.0e6, 'emissivity': 0.92},  # asfalt
            2: {'albedo': 0.20, 'heat_cap': 2.2e6, 'emissivity': 0.90},  # beton
            3: {'albedo': 0.35, 'heat_cap': 1.5e6, 'emissivity': 0.95},  # dach
            4: {'albedo': 0.25, 'heat_cap': 2.5e6, 'emissivity': 0.93},  # trawa
            5: {'albedo': 0.18, 'heat_cap': 3.0e6, 'emissivity': 0.96},  # drzewa
        }
        
        with rasterio.open(landcover_path) as src:
            landcover = src.read(1)
            
        nx, ny = resolution
        self.albedo_map = np.zeros((nx, ny), dtype=np.float32)
        self.heat_capacity_map = np.zeros((nx, ny), dtype=np.float32)
        
        for lc_class, props in landcover_properties.items():
            mask = landcover == lc_class
            self.albedo_map[mask] = props['albedo']
            self.heat_capacity_map[mask] = props['heat_cap']
    
    @jit(nopython=True)
    def compute_solar_radiation(self, timestamp: datetime, cloud_cover: float = 0.2) -> np.ndarray:
        """Obliczenie rozkładu promieniowania słonecznego"""
        solar_position = self.location.get_solarposition(timestamp)
        zenith = solar_position['zenith'].values[0]
        azimuth = solar_position['azimuth'].values[0]
        
        if zenith > 90:  # Noc
            return np.zeros_like(self.albedo_map)
        
        # Model promieniowania bezpośredniego
        dni = 900 * np.cos(np.radians(zenith)) * (1 - cloud_cover)  # W/m²
        
        # Cieniowanie przez budynki - uproszczone
        shadow_map = self._compute_shadows(zenith, azimuth)
        
        radiation = dni * (1 - self.albedo_map) * shadow_map
        return radiation.astype(np.float32)
    
    def compute_surface_temperature(self, radiation: np.ndarray, air_temp: float, 
                                   wind_speed: np.ndarray) -> np.ndarray:
        """Model temperatury powierzchni - bilans energetyczny"""
        stefan_boltzmann = 5.67e-8
        
        # Konwekcja wymuszona
        h_conv = 10.45 - wind_speed + 10 * np.sqrt(wind_speed)  # W/(m²·K)
        
        # Iteracyjne rozwiązanie bilansu
        T_surface = np.full_like(radiation, air_temp + 273.15)
        
        for _ in range(10):  # Iteracje Newton-Raphson
            Q_rad_out = stefan_boltzmann * 0.95 * T_surface**4
            Q_conv = h_conv * (T_surface - air_temp - 273.15)
            
            residual = radiation - Q_rad_out - Q_conv
            dF_dT = -4 * stefan_boltzmann * 0.95 * T_surface**3 - h_conv
            
            T_surface -= residual / (dF_dT + 1e-6)
            
        return (T_surface - 273.15).astype(np.float32)
    
    def compute_heat_island_intensity(self, T_surface: np.ndarray) -> float:
        """Intensywność miejskiej wyspy ciepła"""
        urban_mask = self.albedo_map < 0.2  # Powierzchnie miejskie
        rural_mask = self.albedo_map > 0.3  # Tereny zielone
        
        if urban_mask.sum() > 0 and rural_mask.sum() > 0:
            uhi = np.mean(T_surface[urban_mask]) - np.mean(T_surface[rural_mask])
            return float(uhi)
        return 0.0

# api_backend.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import asyncio
import requests
from datetime import datetime, timedelta
import pickle
import gzip

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # GitHub Pages
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cache symulacji
simulation_cache = {}

class SimulationRequest(BaseModel):
    timestamp: datetime
    domain_bounds: dict  # {"lat_min": 52.0, "lat_max": 52.1, ...}
    resolution: float = 5.0

class WeatherData(BaseModel):
    wind_speed: float
    wind_direction: float
    temperature: float
    humidity: float
    pressure: float

async def fetch_weather_data() -> WeatherData:
    """Pobieranie danych z OpenWeatherMap API"""
    api_key = "YOUR_API_KEY"  # Z secrets w Colab
    lat, lon = 54.156, 19.404  # Elbląg
    
    url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={api_key}"
    response = requests.get(url)
    data = response.json()
    
    return WeatherData(
        wind_speed=data['wind']['speed'],
        wind_direction=data['wind']['deg'],
        temperature=data['main']['temp'] - 273.15,
        humidity=data['main']['humidity'],
        pressure=data['main']['pressure']
    )

@app.post("/api/simulate")
async def run_simulation(request: SimulationRequest):
    """Endpoint głównej symulacji"""
    cache_key = f"{request.timestamp}_{request.resolution}"
    
    if cache_key in simulation_cache:
        age = datetime.now() - simulation_cache[cache_key]['timestamp']
        if age < timedelta(hours=2):
            return simulation_cache[cache_key]['data']
    
    try:
        # Pobierz dane pogodowe
        weather = await fetch_weather_data()
        
        # Inicjalizacja solverów
        solver = WindFieldSolver((200, 200, 50), request.resolution)
        thermal = UrbanThermalModel(54.156, 19.404)
        
        # Załaduj geometrię z Google Drive
        solver.load_geometry(
            '/content/drive/MyDrive/microclimate/buildings.tif',
            '/content/drive/MyDrive/microclimate/dem.tif',
            '/content/drive/MyDrive/microclimate/canopy.tif'
        )
        
        # Obliczenia przepływu
        u_inlet = weather.wind_speed * np.cos(np.radians(weather.wind_direction))
        v_inlet = weather.wind_speed * np.sin(np.radians(weather.wind_direction))
        
        phi = solver.solve_potential_flow(u_inlet, v_inlet)
        velocity = solver.compute_velocity_field(phi)
        turbulence = solver.compute_turbulence_intensity(velocity)
        
        # Model termiczny
        thermal.load_surface_properties('/content/drive/MyDrive/microclimate/landcover.tif', (200, 200))
        radiation = thermal.compute_solar_radiation(request.timestamp)
        T_surface = thermal.compute_surface_temperature(
            radiation, weather.temperature, velocity['speed'][:,:,2]  # 10m wysokość
        )
        uhi = thermal.compute_heat_island_intensity(T_surface)
        
        # Kompresja wyników
        result = {
            'timestamp': request.timestamp.isoformat(),
            'weather': weather.dict(),
            'wind_field': {
                'u': velocity['u'][:,:,2].tolist(),  # Tylko warstwa 10m
                'v': velocity['v'][:,:,2].tolist(),
                'speed': velocity['speed'][:,:,2].tolist(),
                'turbulence': turbulence[:,:,2].tolist()
            },
            'thermal': {
                'surface_temperature': T_surface.tolist(),
                'uhi_intensity': uhi
            },
            'metadata': {
                'resolution': request.resolution,
                'domain_size': [200, 200],
                'computation_time': 0  # Uzupełnić
            }
        }
        
        # Cache
        simulation_cache[cache_key] = {
            'timestamp': datetime.now(),
            'data': result
        }
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/status")
async def get_status():
    """Status systemu"""
    return {
        'status': 'operational',
        'cache_size': len(simulation_cache),
        'last_update': datetime.now().isoformat()
    }

# Automatyczne uruchomienie co 2h
async def auto_simulation_task():
    while True:
        await asyncio.sleep(7200)  # 2 godziny
        request = SimulationRequest(
            timestamp=datetime.now(),
            domain_bounds={'lat_min': 54.1, 'lat_max': 54.2, 'lon_min': 19.3, 'lon_max': 19.5},
            resolution=5.0
        )
        await run_simulation(request)

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(auto_simulation_task())

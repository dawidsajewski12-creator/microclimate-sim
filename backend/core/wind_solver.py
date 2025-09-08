# wind_solver.py
import numpy as np
from scipy import ndimage, interpolate
from numba import jit, prange
import rasterio
from typing import Tuple, Dict
import logging

class WindFieldSolver:
    """Solver przepływu potencjalnego z korekcją dla przeszkód miejskich"""
    
    def __init__(self, domain_size: Tuple[int, int, int], resolution: float = 5.0):
        self.nx, self.ny, self.nz = domain_size
        self.dx = resolution  # m/piksel
        self.obstacles = np.zeros(domain_size, dtype=np.float32)
        self.roughness = np.ones((self.nx, self.ny), dtype=np.float32) * 0.03  # z0 default
        
    def load_geometry(self, buildings_path: str, dem_path: str, canopy_path: str):
        """Ładowanie geometrii z rastrów GIS"""
        with rasterio.open(dem_path) as src:
            dem = src.read(1).astype(np.float32)
            self.dem = ndimage.zoom(dem, (self.nx/dem.shape[0], self.ny/dem.shape[1]))
            
        with rasterio.open(buildings_path) as src:
            buildings = src.read(1).astype(np.float32)
            buildings = ndimage.zoom(buildings, (self.nx/buildings.shape[0], self.ny/buildings.shape[1]))
            
        # Konwersja do siatki 3D przeszkód
        for i in range(self.nx):
            for j in range(self.ny):
                height_idx = int((self.dem[i,j] + buildings[i,j]) / self.dx)
                if height_idx < self.nz:
                    self.obstacles[i, j, :height_idx] = 1.0
                    
        # Aktualizacja szorstkości na podstawie pokrycia terenu
        with rasterio.open(canopy_path) as src:
            canopy = src.read(1).astype(np.float32)
            canopy = ndimage.zoom(canopy, (self.nx/canopy.shape[0], self.ny/canopy.shape[1]))
            self.roughness = 0.03 + 0.5 * (canopy / canopy.max())  # z0 = 0.03-0.53m
    
    @jit(nopython=True, parallel=True)
    def solve_potential_flow(self, u_inlet: float, v_inlet: float, 
                           max_iter: int = 1000, tol: float = 1e-4) -> np.ndarray:
        """Rozwiązanie równania Laplace'a dla potencjału prędkości"""
        phi = np.zeros((self.nx, self.ny, self.nz), dtype=np.float32)
        phi_new = phi.copy()
        
        # Warunki brzegowe - profil logarytmiczny wiatru
        for k in prange(self.nz):
            z = k * self.dx
            if z > 0:
                u_profile = u_inlet * np.log(z/0.03) / np.log(100/0.03)
                phi[0, :, k] = u_profile * self.dx
        
        # Iteracyjne rozwiązanie SOR
        omega = 1.8  # współczynnik relaksacji
        for iteration in range(max_iter):
            for i in prange(1, self.nx-1):
                for j in prange(1, self.ny-1):
                    for k in prange(1, self.nz-1):
                        if self.obstacles[i,j,k] < 0.5:
                            phi_new[i,j,k] = (1-omega)*phi[i,j,k] + omega/6 * (
                                phi[i+1,j,k] + phi[i-1,j,k] +
                                phi[i,j+1,k] + phi[i,j-1,k] +
                                phi[i,j,k+1] + phi[i,j,k-1]
                            )
            
            # Sprawdzenie zbieżności
            residual = np.max(np.abs(phi_new - phi))
            phi = phi_new.copy()
            if residual < tol:
                break
                
        return phi
    
    def compute_velocity_field(self, phi: np.ndarray) -> Dict[str, np.ndarray]:
        """Obliczenie pola prędkości z potencjału"""
        u = -np.gradient(phi, axis=0) / self.dx
        v = -np.gradient(phi, axis=1) / self.dx
        w = -np.gradient(phi, axis=2) / self.dx
        
        # Korekcja dla warstwy przyziemnej
        for k in range(min(10, self.nz)):
            z = k * self.dx
            if z > 0:
                factor = np.log(z/self.roughness) / np.log(10/self.roughness)
                factor = np.clip(factor, 0, 1)
                u[:,:,k] *= factor
                v[:,:,k] *= factor
        
        # Zerowanie w przeszkodach
        u[self.obstacles > 0.5] = 0
        v[self.obstacles > 0.5] = 0
        w[self.obstacles > 0.5] = 0
        
        speed = np.sqrt(u**2 + v**2 + w**2)
        
        return {
            'u': u.astype(np.float16),  # Kompresja pamięci
            'v': v.astype(np.float16),
            'w': w.astype(np.float16),
            'speed': speed.astype(np.float16)
        }
    
    def compute_turbulence_intensity(self, velocity: Dict[str, np.ndarray]) -> np.ndarray:
        """Model turbulencji k-epsilon uproszczony"""
        speed = velocity['speed']
        
        # TKE na podstawie gradientów prędkości
        du_dx = np.gradient(velocity['u'], axis=0)
        dv_dy = np.gradient(velocity['v'], axis=1)
        dw_dz = np.gradient(velocity['w'], axis=2)
        
        strain_rate = np.sqrt(du_dx**2 + dv_dy**2 + dw_dz**2)
        tke = 0.1 * strain_rate * speed  # Uproszczony model
        
        # Zwiększona turbulencja za przeszkodami
        wake_mask = ndimage.binary_dilation(self.obstacles > 0.5, iterations=3)
        tke[wake_mask] *= 2.5
        
        turbulence_intensity = np.sqrt(2*tke/3) / (speed + 0.1)
        return np.clip(turbulence_intensity, 0, 1).astype(np.float16)

# backend/core/data_processor.py
"""
Główny moduł przetwarzania i integracji danych GIS
"""

import numpy as np
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from scipy import ndimage
import logging
from typing import Dict, Tuple, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

class UrbanDataProcessor:
    """Procesor danych miejskich - integracja wszystkich warstw GIS"""
    
    def __init__(self, data_dir: str, resolution: float = 5.0):
        self.data_dir = Path(data_dir)
        self.resolution = resolution
        self.data_layers = {}
        self.metadata = {}
        self.bounds = None
        self.shape = None
        
    def load_all_layers(self) -> Dict[str, np.ndarray]:
        """Wczytaj wszystkie warstwy danych"""
        
        required_layers = {
            'buildings': 'buildings.tif',
            'dem': 'dem.tif', 
            'landcover': 'landcover.tif',
            'tree_canopy': 'tree_canopy.tif'
        }
        
        for layer_name, filename in required_layers.items():
            filepath = self.data_dir / filename
            
            if not filepath.exists():
                logger.warning(f"Brak pliku {filename}, tworzenie warstwy zastępczej...")
                self._create_placeholder_layer(layer_name)
            else:
                self._load_layer(layer_name, filepath)
        
        # Normalizacja do wspólnej siatki
        self._normalize_grids()
        
        return self.data_layers
    
    def _load_layer(self, name: str, filepath: Path):
        """Wczytaj pojedynczą warstwę"""
        logger.info(f"Ładowanie warstwy {name} z {filepath}")
        
        with rasterio.open(filepath) as src:
            data = src.read(1).astype(np.float32)
            
            # Zapisz metadane
            self.metadata[name] = {
                'crs': src.crs,
                'transform': src.transform,
                'bounds': src.bounds,
                'shape': data.shape,
                'nodata': src.nodata
            }
            
            # Ustaw wspólne granice na podstawie pierwszej warstwy
            if self.bounds is None:
                self.bounds = src.bounds
                self.shape = data.shape
            
            # Wypełnij nodata
            if src.nodata is not None:
                data[data == src.nodata] = 0
            
            self.data_layers[name] = data
            
            # Statystyki
            logger.info(f"  Wymiary: {data.shape}")
            logger.info(f"  Zakres: [{np.min(data):.1f}, {np.max(data):.1f}]")
    
    def _create_placeholder_layer(self, name: str):
        """Utwórz zastępczą warstwę jeśli brak danych"""
        
        if self.shape is None:
            # Domyślny rozmiar siatki
            self.shape = (200, 200)
        
        if name == 'dem':
            # Płaski teren z małym szumem
            data = np.random.uniform(0, 2, self.shape).astype(np.float32)
        elif name == 'tree_canopy':
            # Losowe skupiska drzew
            data = np.zeros(self.shape, dtype=np.float32)
            # Dodaj kilka skupisk
            for _ in range(10):
                x, y = np.random.randint(0, self.shape[0]), np.random.randint(0, self.shape[1])
                radius = np.random.randint(5, 20)
                y_grid, x_grid = np.ogrid[:self.shape[0], :self.shape[1]]
                mask = (x_grid - x)**2 + (y_grid - y)**2 <= radius**2
                data[mask] = np.random.uniform(5, 15)  # Wysokość drzew
        elif name == 'landcover':
            # Domyślne klasy pokrycia
            data = np.ones(self.shape, dtype=np.float32)  # Klasa 1 = asfalt
            # Dodaj parki
            for _ in range(5):
                x, y = np.random.randint(0, self.shape[0]), np.random.randint(0, self.shape[1])
                radius = np.random.randint(10, 30)
                y_grid, x_grid = np.ogrid[:self.shape[0], :self.shape[1]]
                mask = (x_grid - x)**2 + (y_grid - y)**2 <= radius**2
                data[mask] = 4  # Klasa 4 = trawa
        else:
            data = np.zeros(self.shape, dtype=np.float32)
        
        self.data_layers[name] = data
        logger.warning(f"Utworzono warstwę zastępczą dla {name}")
    
    def _normalize_grids(self):
        """Normalizuj wszystkie warstwy do wspólnej siatki"""
        
        if not self.data_layers:
            return
        
        # Znajdź docelowe wymiary (największa warstwa)
        target_shape = max([layer.shape for layer in self.data_layers.values()])
        
        for name, data in self.data_layers.items():
            if data.shape != target_shape:
                logger.info(f"Skalowanie {name} z {data.shape} do {target_shape}")
                zoom_factors = (target_shape[0] / data.shape[0], 
                               target_shape[1] / data.shape[1])
                self.data_layers[name] = ndimage.zoom(data, zoom_factors, order=1)
    
    def create_urban_geometry(self) -> Dict[str, np.ndarray]:
        """Utwórz geometrię 3D miasta dla symulacji"""
        
        buildings = self.data_layers.get('buildings', np.zeros(self.shape))
        dem = self.data_layers.get('dem', np.zeros(self.shape))
        trees = self.data_layers.get('tree_canopy', np.zeros(self.shape))
        
        # Geometria dla solvera CFD
        geometry = {
            'surface_elevation': dem,
            'building_height': buildings,
            'tree_height': trees,
            'total_height': dem + buildings + trees,
            'roughness_length': self._calculate_roughness()
        }
        
        return geometry
    
    def _calculate_roughness(self) -> np.ndarray:
        """Oblicz długość szorstkości aerodynamicznej z0"""
        
        landcover = self.data_layers.get('landcover', np.ones(self.shape))
        buildings = self.data_layers.get('buildings', np.zeros(self.shape))
        trees = self.data_layers.get('tree_canopy', np.zeros(self.shape))
        
        # Mapowanie klas pokrycia na z0 [m]
        z0_map = {
            1: 0.01,   # Woda
            2: 0.03,   # Asfalt/beton
            3: 0.05,   # Niskie budynki
            4: 0.10,   # Trawa
            5: 0.30,   # Krzewy
            6: 0.50,   # Drzewa liściaste
            7: 0.80,   # Drzewa iglaste
            8: 1.00,   # Gęsta zabudowa
        }
        
        z0 = np.zeros_like(landcover)
        
        for lc_class, roughness in z0_map.items():
            mask = landcover == lc_class
            z0[mask] = roughness
        
        # Modyfikacja dla budynków (metoda MacDonald)
        building_mask = buildings > 0
        if building_mask.any():
            # Frontal area density
            lambda_f = np.zeros_like(buildings)
            for i in range(3, buildings.shape[0]-3):
                for j in range(3, buildings.shape[1]-3):
                    if buildings[i,j] > 0:
                        # Lokalna gęstość zabudowy
                        local_area = buildings[i-3:i+4, j-3:j+4]
                        lambda_f[i,j] = np.sum(local_area > 0) / 49.0
            
            # z0 dla obszarów miejskich
            z0[building_mask] = 0.1 * buildings[building_mask] * (1 - lambda_f[building_mask]**0.5)
            z0[building_mask] = np.clip(z0[building_mask], 0.1, 2.0)
        
        # Modyfikacja dla drzew
        tree_mask = trees > 1
        z0[tree_mask] = 0.1 * trees[tree_mask]
        
        return z0
    
    def calculate_sky_view_factor(self) -> np.ndarray:
        """Oblicz Sky View Factor (SVF) dla modelu termicznego"""
        
        buildings = self.data_layers.get('buildings', np.zeros(self.shape))
        trees = self.data_layers.get('tree_canopy', np.zeros(self.shape))
        
        # Uproszczony SVF oparty na lokalnej wysokości przeszkód
        svf = np.ones(self.shape, dtype=np.float32)
        
        # Kernel do analizy sąsiedztwa (8 kierunków)
        angles = np.linspace(0, 2*np.pi, 16, endpoint=False)
        
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                max_angles = []
                
                for angle in angles:
                    # Sprawdź przeszkody w danym kierunku
                    for dist in range(1, 20):  # Do 100m (20 * 5m)
                        di = int(dist * np.sin(angle))
                        dj = int(dist * np.cos(angle))
                        
                        if (0 <= i+di < self.shape[0] and 
                            0 <= j+dj < self.shape[1]):
                            
                            obstacle_height = buildings[i+di, j+dj] + trees[i+di, j+dj]
                            if obstacle_height > 0:
                                # Kąt horyzontu
                                horizon_angle = np.arctan(obstacle_height / (dist * self.resolution))
                                max_angles.append(horizon_angle)
                                break
                    else:
                        max_angles.append(0)  # Brak przeszkód
                
                # SVF = średnia z cos(kąt_horyzontu)
                if max_angles:
                    svf[i, j] = np.mean([np.cos(a) for a in max_angles])
        
        return svf
    
    def export_for_simulation(self, output_dir: str):
        """Eksportuj przetworzone dane dla symulacji"""
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Zapisz wszystkie warstwy
        for name, data in self.data_layers.items():
            filepath = output_path / f"{name}_processed.npy"
            np.save(filepath, data)
            logger.info(f"Zapisano {name} do {filepath}")
        
        # Zapisz geometrię
        geometry = self.create_urban_geometry()
        for name, data in geometry.items():
            filepath = output_path / f"geometry_{name}.npy"
            np.save(filepath, data)
        
        # Zapisz SVF
        svf = self.calculate_sky_view_factor()
        np.save(output_path / "svf.npy", svf)
        
        # Zapisz metadane
        import json
        metadata = {
            'resolution': self.resolution,
            'shape': self.shape,
            'bounds': self.bounds,
            'layers': list(self.data_layers.keys())
        }
        
        with open(output_path / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Eksport zakończony: {output_path}")
    
    def validate_data(self) -> Dict[str, bool]:
        """Walidacja kompletności i jakości danych"""
        
        validation_results = {}
        
        for name, data in self.data_layers.items():
            checks = {
                'exists': data is not None,
                'not_empty': data.size > 0 if data is not None else False,
                'has_values': np.any(data > 0) if data is not None else False,
                'no_infinite': not np.any(np.isinf(data)) if data is not None else False,
                'reasonable_range': True  # Sprawdzenie zakresu wartości
            }
            
            # Specyficzne sprawdzenia zakresu
            if name == 'buildings' and data is not None:
                checks['reasonable_range'] = np.all(data[data > 0] < 200)  # Max 200m wysokości
            elif name == 'dem' and data is not None:
                checks['reasonable_range'] = np.all((data >= -100) & (data < 3000))  # -100 do 3000m npm
            elif name == 'tree_canopy' and data is not None:
                checks['reasonable_range'] = np.all(data[data > 0] < 50)  # Max 50m wysokości
            
            validation_results[name] = all(checks.values())
            
            if not validation_results[name]:
                logger.warning(f"Walidacja {name} nieudana: {checks}")
        
        return validation_results


# Funkcja testowa
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    processor = UrbanDataProcessor(
        data_dir="/content/drive/MyDrive/microclimate/data/processed",
        resolution=5.0
    )
    
    # Wczytaj dane
    processor.load_all_layers()
    
    # Walidacja
    validation = processor.validate_data()
    print(f"Walidacja danych: {validation}")
    
    # Eksport
    processor.export_for_simulation("/content/drive/MyDrive/microclimate/simulation_data")

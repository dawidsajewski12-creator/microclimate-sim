# backend/utils/gpkg_converter.py
"""
Konwerter GeoPackage (budynki wektorowe) do rastra wysokości
"""

import geopandas as gpd
import rasterio
from rasterio import features
from rasterio.transform import from_bounds
import numpy as np
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class BuildingsRasterizer:
    """Konwersja budynków z GPKG do rastra wysokości"""
    
    def __init__(self, resolution: float = 5.0):
        self.resolution = resolution  # metry/piksel
        self.buildings_gdf = None
        self.bounds = None
        
    def load_buildings(self, gpkg_path: str, height_field: str = 'height'):
        """
        Wczytaj budynki z GeoPackage
        
        Args:
            gpkg_path: Ścieżka do pliku .gpkg
            height_field: Nazwa pola z wysokością budynku (domyślnie 'height')
        """
        logger.info(f"Ładowanie budynków z {gpkg_path}")
        
        # Wczytaj GeoPackage
        self.buildings_gdf = gpd.read_file(gpkg_path)
        
        # Sprawdź czy istnieje pole wysokości
        if height_field not in self.buildings_gdf.columns:
            logger.warning(f"Brak pola '{height_field}', szacowanie wysokości...")
            # Oszacuj wysokość na podstawie liczby kondygnacji lub ustaw domyślną
            if 'floors' in self.buildings_gdf.columns:
                self.buildings_gdf[height_field] = self.buildings_gdf['floors'] * 3.5  # 3.5m/piętro
            elif 'levels' in self.buildings_gdf.columns:
                self.buildings_gdf[height_field] = self.buildings_gdf['levels'] * 3.5
            else:
                # Domyślna wysokość 10m
                self.buildings_gdf[height_field] = 10.0
                logger.warning("Używam domyślnej wysokości 10m dla wszystkich budynków")
        
        self.height_field = height_field
        
        # Reprojekcja do układu metrycznego jeśli potrzeba
        if self.buildings_gdf.crs.to_epsg() == 4326:  # WGS84
            logger.info("Reprojekcja z WGS84 do UTM")
            # Automatyczny wybór strefy UTM
            lon = self.buildings_gdf.geometry.centroid.x.mean()
            utm_zone = int((lon + 180) / 6) + 1
            utm_crs = f"EPSG:326{utm_zone:02d}" if self.buildings_gdf.geometry.centroid.y.mean() >= 0 else f"EPSG:327{utm_zone:02d}"
            self.buildings_gdf = self.buildings_gdf.to_crs(utm_crs)
        
        # Oblicz granice
        self.bounds = self.buildings_gdf.total_bounds  # [minx, miny, maxx, maxy]
        
        logger.info(f"Załadowano {len(self.buildings_gdf)} budynków")
        logger.info(f"Zakres wysokości: {self.buildings_gdf[height_field].min():.1f} - {self.buildings_gdf[height_field].max():.1f}m")
        
    def rasterize(self, output_path: str, bounds: Optional[Tuple] = None) -> np.ndarray:
        """
        Konwertuj budynki do rastra
        
        Args:
            output_path: Ścieżka wyjściowa dla rastra .tif
            bounds: Opcjonalne granice [minx, miny, maxx, maxy]
        
        Returns:
            numpy array z wysokościami budynków
        """
        if self.buildings_gdf is None:
            raise ValueError("Najpierw załaduj budynki używając load_buildings()")
        
        # Użyj podanych granic lub obliczonych
        if bounds is None:
            bounds = self.bounds
        
        # Oblicz wymiary rastra
        width = int((bounds[2] - bounds[0]) / self.resolution)
        height = int((bounds[3] - bounds[1]) / self.resolution)
        
        logger.info(f"Tworzenie rastra {width}x{height} pikseli @ {self.resolution}m/piksel")
        
        # Utwórz transformację
        transform = from_bounds(bounds[0], bounds[1], bounds[2], bounds[3], width, height)
        
        # Przygotuj pary (geometria, wysokość)
        shapes = [(geom, value) for geom, value in 
                  zip(self.buildings_gdf.geometry, self.buildings_gdf[self.height_field])]
        
        # Rasteryzacja
        raster = features.rasterize(
            shapes,
            out_shape=(height, width),
            transform=transform,
            fill=0,  # Wartość tła (brak budynku)
            dtype=rasterio.float32
        )
        
        # Zapisz jako GeoTIFF
        profile = {
            'driver': 'GTiff',
            'height': height,
            'width': width,
            'count': 1,
            'dtype': rasterio.float32,
            'crs': self.buildings_gdf.crs,
            'transform': transform,
            'compress': 'lzw',
            'nodata': 0
        }
        
        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(raster, 1)
        
        logger.info(f"Zapisano raster do {output_path}")
        
        # Statystyki
        building_pixels = (raster > 0).sum()
        coverage = building_pixels / (width * height) * 100
        logger.info(f"Pokrycie budynkami: {coverage:.1f}%")
        logger.info(f"Średnia wysokość: {raster[raster > 0].mean():.1f}m")
        
        return raster
    
    def create_building_mask(self, output_path: str) -> np.ndarray:
        """Utwórz binarną maskę budynków"""
        raster = self.rasterize(output_path.replace('.tif', '_temp.tif'))
        mask = (raster > 0).astype(np.uint8)
        
        # Zapisz maskę
        with rasterio.open(output_path.replace('.tif', '_temp.tif')) as src:
            profile = src.profile
            profile['dtype'] = rasterio.uint8
            profile['nodata'] = None
            
            with rasterio.open(output_path, 'w', **profile) as dst:
                dst.write(mask, 1)
        
        return mask
    
    def analyze_urban_morphology(self) -> dict:
        """Analiza morfologii miejskiej"""
        if self.buildings_gdf is None:
            raise ValueError("Najpierw załaduj budynki")
        
        stats = {
            'building_count': len(self.buildings_gdf),
            'total_footprint_m2': self.buildings_gdf.geometry.area.sum(),
            'avg_building_area_m2': self.buildings_gdf.geometry.area.mean(),
            'avg_height_m': self.buildings_gdf[self.height_field].mean(),
            'max_height_m': self.buildings_gdf[self.height_field].max(),
            'total_volume_m3': (self.buildings_gdf.geometry.area * self.buildings_gdf[self.height_field]).sum(),
        }
        
        # Klasyfikacja wysokości
        height_classes = {
            'low_rise': (self.buildings_gdf[self.height_field] <= 10).sum(),  # ≤10m
            'mid_rise': ((self.buildings_gdf[self.height_field] > 10) & 
                        (self.buildings_gdf[self.height_field] <= 25)).sum(),  # 10-25m
            'high_rise': (self.buildings_gdf[self.height_field] > 25).sum()  # >25m
        }
        
        stats['height_distribution'] = height_classes
        
        return stats


# Funkcja pomocnicza do przetwarzania wszystkich danych
def preprocess_all_data(data_dir: str, output_dir: str):
    """
    Przetwórz wszystkie dane GIS do wspólnego formatu
    
    Args:
        data_dir: Folder z danymi źródłowymi
        output_dir: Folder na przetworzone rastry
    """
    import os
    from pathlib import Path
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Konwertuj budynki z GPKG
    logger.info("=== Przetwarzanie budynków ===")
    rasterizer = BuildingsRasterizer(resolution=5.0)
    
    gpkg_path = os.path.join(data_dir, 'buildings.gpkg')
    if os.path.exists(gpkg_path):
        rasterizer.load_buildings(gpkg_path)
        
        # Zapisz raster wysokości
        buildings_raster_path = os.path.join(output_dir, 'buildings.tif')
        rasterizer.rasterize(buildings_raster_path)
        
        # Analiza morfologii
        morphology = rasterizer.analyze_urban_morphology()
        logger.info(f"Morfologia miejska: {morphology}")
        
        # Użyj granic budynków jako referencji dla innych rastrów
        reference_bounds = rasterizer.bounds
    else:
        logger.error(f"Nie znaleziono pliku {gpkg_path}")
        return
    
    # 2. Dopasuj pozostałe rastry do tej samej siatki
    logger.info("=== Dopasowanie pozostałych rastrów ===")
    
    raster_files = ['dem.tif', 'landcover.tif', 'tree_canopy.tif']
    
    for raster_file in raster_files:
        input_path = os.path.join(data_dir, raster_file)
        output_path = os.path.join(output_dir, raster_file)
        
        if not os.path.exists(input_path):
            logger.warning(f"Pominięto {raster_file} - plik nie istnieje")
            continue
        
        logger.info(f"Przetwarzanie {raster_file}")
        
        # Wczytaj raster źródłowy
        with rasterio.open(input_path) as src:
            # Oblicz docelowe wymiary
            dst_width = int((reference_bounds[2] - reference_bounds[0]) / 5.0)
            dst_height = int((reference_bounds[3] - reference_bounds[1]) / 5.0)
            
            # Transformacja docelowa
            dst_transform = from_bounds(
                reference_bounds[0], reference_bounds[1],
                reference_bounds[2], reference_bounds[3],
                dst_width, dst_height
            )
            
            # Przygotuj array docelowy
            dst_array = np.zeros((dst_height, dst_width), dtype=np.float32)
            
            # Reprojekcja
            from rasterio.warp import reproject, Resampling
            
            reproject(
                source=rasterio.band(src, 1),
                destination=dst_array,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=dst_transform,
                dst_crs=rasterizer.buildings_gdf.crs,
                resampling=Resampling.bilinear
            )
            
            # Zapisz
            profile = {
                'driver': 'GTiff',
                'height': dst_height,
                'width': dst_width,
                'count': 1,
                'dtype': rasterio.float32,
                'crs': rasterizer.buildings_gdf.crs,
                'transform': dst_transform,
                'compress': 'lzw'
            }
            
            with rasterio.open(output_path, 'w', **profile) as dst:
                dst.write(dst_array, 1)
            
            logger.info(f"Zapisano {output_path}")
    
    logger.info("=== Przetwarzanie zakończone ===")
    

if __name__ == "__main__":
    # Test konwersji
    logging.basicConfig(level=logging.INFO)
    
    # Przykład użycia
    preprocess_all_data(
        data_dir="/content/drive/MyDrive/microclimate/data/raw",
        output_dir="/content/drive/MyDrive/microclimate/data/processed"
    )

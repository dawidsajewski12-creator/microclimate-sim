# backend/utils/cache_manager.py
"""
Manager cache z kompresją i optymalizacją pamięci
"""

import os
import json
import hashlib
import pickle
import gzip
import time
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Any, Dict, Optional, Tuple
import logging
import joblib

logger = logging.getLogger(__name__)

class CacheManager:
    """Zarządzanie cache symulacji z kompresją"""
    
    def __init__(self, cache_dir: str = "/content/drive/MyDrive/microclimate/cache", 
                 ttl_hours: float = 2.0,
                 max_size_mb: float = 500.0):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ttl = timedelta(hours=ttl_hours)
        self.max_size_mb = max_size_mb
        self.metadata_file = self.cache_dir / "cache_metadata.json"
        self.metadata = self._load_metadata()
        
    def _load_metadata(self) -> Dict:
        """Wczytaj metadane cache"""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_metadata(self):
        """Zapisz metadane cache"""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)
    
    def get_key(self, params: Dict) -> str:
        """Generuj unikalny klucz dla parametrów"""
        # Sortuj i serializuj parametry
        param_str = json.dumps(params, sort_keys=True, default=str)
        return hashlib.sha256(param_str.encode()).hexdigest()[:16]
    
    def _compress_data(self, data: Dict) -> bytes:
        """Kompresuj dane do binarnego formatu"""
        # Konwertuj numpy arrays do list dla serializacji
        compressed_data = {}
        
        for key, value in data.items():
            if isinstance(value, np.ndarray):
                # Kompresja array - float16 dla oszczędności
                compressed_data[key] = {
                    'type': 'ndarray',
                    'shape': value.shape,
                    'dtype': str(value.dtype),
                    'data': value.astype(np.float16).tobytes()
                }
            elif isinstance(value, dict):
                compressed_data[key] = self._compress_nested_dict(value)
            else:
                compressed_data[key] = value
        
        # Pickle + gzip
        return gzip.compress(pickle.dumps(compressed_data, protocol=4))
    
    def _compress_nested_dict(self, d: Dict) -> Dict:
        """Rekursywna kompresja zagnieżdżonych słowników"""
        result = {}
        for key, value in d.items():
            if isinstance(value, np.ndarray):
                result[key] = {
                    'type': 'ndarray',
                    'shape': value.shape,
                    'dtype': str(value.dtype),
                    'data': value.astype(np.float16).tobytes()
                }
            elif isinstance(value, dict):
                result[key] = self._compress_nested_dict(value)
            else:
                result[key] = value
        return result
    
    def _decompress_data(self, compressed: bytes) -> Dict:
        """Dekompresuj dane"""
        data = pickle.loads(gzip.decompress(compressed))
        return self._restore_arrays(data)
    
    def _restore_arrays(self, data: Dict) -> Dict:
        """Przywróć numpy arrays z kompresji"""
        result = {}
        for key, value in data.items():
            if isinstance(value, dict):
                if value.get('type') == 'ndarray':
                    # Przywróć array
                    arr = np.frombuffer(value['data'], dtype=np.float16)
                    result[key] = arr.reshape(value['shape']).astype(value['dtype'])
                else:
                    result[key] = self._restore_arrays(value)
            else:
                result[key] = value
        return result
    
    def save(self, key: str, data: Dict, metadata: Optional[Dict] = None):
        """Zapisz dane do cache"""
        file_path = self.cache_dir / f"{key}.cache"
        
        try:
            # Kompresuj i zapisz
            compressed = self._compress_data(data)
            
            with open(file_path, 'wb') as f:
                f.write(compressed)
            
            # Aktualizuj metadane
            file_size_mb = len(compressed) / (1024 * 1024)
            
            self.metadata[key] = {
                'timestamp': datetime.now().isoformat(),
                'size_mb': round(file_size_mb, 2),
                'metadata': metadata or {}
            }
            
            self._save_metadata()
            
            logger.info(f"Cache zapisany: {key} ({file_size_mb:.2f}MB)")
            
            # Sprawdź limit rozmiaru
            self._check_size_limit()
            
        except Exception as e:
            logger.error(f"Błąd zapisu cache {key}: {e}")
    
    def load(self, key: str) -> Optional[Dict]:
        """Wczytaj dane z cache"""
        file_path = self.cache_dir / f"{key}.cache"
        
        if not file_path.exists():
            return None
        
        # Sprawdź wiek
        if key in self.metadata:
            timestamp = datetime.fromisoformat(self.metadata[key]['timestamp'])
            age = datetime.now() - timestamp
            
            if age > self.ttl:
                logger.info(f"Cache {key} wygasł ({age.total_seconds()/3600:.1f}h)")
                self.delete(key)
                return None
        
        try:
            with open(file_path, 'rb') as f:
                compressed = f.read()
            
            data = self._decompress_data(compressed)
            logger.info(f"Cache wczytany: {key}")
            return data
            
        except Exception as e:
            logger.error(f"Błąd odczytu cache {key}: {e}")
            return None
    
    def delete(self, key: str):
        """Usuń wpis z cache"""
        file_path = self.cache_dir / f"{key}.cache"
        
        if file_path.exists():
            file_path.unlink()
        
        if key in self.metadata:
            del self.metadata[key]
            self._save_metadata()
        
        logger.info(f"Cache usunięty: {key}")
    
    def clear_expired(self):
        """Usuń wygasłe wpisy"""
        now = datetime.now()
        to_delete = []
        
        for key, info in self.metadata.items():
            timestamp = datetime.fromisoformat(info['timestamp'])
            age = now - timestamp
            
            if age > self.ttl:
                to_delete.append(key)
        
        for key in to_delete:
            self.delete(key)
        
        if to_delete:
            logger.info(f"Usunięto {len(to_delete)} wygasłych wpisów")
    
    def _check_size_limit(self):
        """Sprawdź i wymuś limit rozmiaru cache"""
        total_size = sum(info['size_mb'] for info in self.metadata.values())
        
        if total_size > self.max_size_mb:
            logger.warning(f"Cache przekroczył limit ({total_size:.1f}/{self.max_size_mb}MB)")
            
            # Usuń najstarsze wpisy
            sorted_keys = sorted(
                self.metadata.keys(),
                key=lambda k: self.metadata[k]['timestamp']
            )
            
            while total_size > self.max_size_mb * 0.8 and sorted_keys:  # Zostaw 20% bufora
                key_to_delete = sorted_keys.pop(0)
                size = self.metadata[key_to_delete]['size_mb']
                self.delete(key_to_delete)
                total_size -= size
    
    def get_stats(self) -> Dict:
        """Pobierz statystyki cache"""
        if not self.metadata:
            return {
                'entries': 0,
                'total_size_mb': 0,
                'oldest': None,
                'newest': None
            }
        
        timestamps = [datetime.fromisoformat(info['timestamp']) 
                     for info in self.metadata.values()]
        
        return {
            'entries': len(self.metadata),
            'total_size_mb': sum(info['size_mb'] for info in self.metadata.values()),
            'oldest': min(timestamps).isoformat() if timestamps else None,
            'newest': max(timestamps).isoformat() if timestamps else None,
            'limit_mb': self.max_size_mb,
            'ttl_hours': self.ttl.total_seconds() / 3600
        }
    
    def exists(self, key: str) -> bool:
        """Sprawdź czy klucz istnieje w cache"""
        if key not in self.metadata:
            return False
        
        # Sprawdź czy nie wygasł
        timestamp = datetime.fromisoformat(self.metadata[key]['timestamp'])
        age = datetime.now() - timestamp
        
        return age <= self.ttl


class ResultsArchive:
    """Archiwum wyników symulacji do analizy historycznej"""
    
    def __init__(self, archive_dir: str = "/content/drive/MyDrive/microclimate/results"):
        self.archive_dir = Path(archive_dir)
        self.archive_dir.mkdir(parents=True, exist_ok=True)
        
    def save_results(self, simulation_id: str, results: Dict, compress: bool = True):
        """Zapisz wyniki symulacji do archiwum"""
        
        # Utwórz folder na podstawie daty
        date_folder = self.archive_dir / datetime.now().strftime("%Y-%m-%d")
        date_folder.mkdir(exist_ok=True)
        
        # Nazwa pliku z timestampem
        timestamp = datetime.now().strftime("%H%M%S")
        filename = f"{simulation_id}_{timestamp}"
        
        if compress:
            # Zapisz jako skompresowany joblib
            file_path = date_folder / f"{filename}.joblib"
            joblib.dump(results, file_path, compress=('gzip', 3))
        else:
            # Zapisz jako JSON (tylko metadane)
            file_path = date_folder / f"{filename}.json"
            
            # Wyodrębnij tylko metadane (bez dużych arrays)
            metadata = {
                'id': results.get('id'),
                'timestamp': results.get('timestamp'),
                'weather': results.get('weather'),
                'metadata': results.get('metadata'),
                'thermal_summary': {
                    'uhi_intensity': results.get('thermal_field', {}).get('uhi_intensity'),
                    'max_temp': float(np.max(results.get('thermal_field', {}).get('surface_temperature', [0]))),
                    'min_temp': float(np.min(results.get('thermal_field', {}).get('surface_temperature', [0])))
                },
                'wind_summary': {
                    'max_speed': float(np.max(results.get('wind_field', {}).get('speed_10m', [0]))),
                    'avg_speed': float(np.mean(results.get('wind_field', {}).get('speed_10m', [0])))
                }
            }
            
            with open(file_path, 'w') as f:
                json.dump(metadata, f, indent=2)
        
        logger.info(f"Wyniki zarchiwizowane: {file_path}")
        
        # Cleanup starych archiwów (>30 dni)
        self._cleanup_old_archives()
    
    def _cleanup_old_archives(self, days: int = 30):
        """Usuń stare archiwa"""
        cutoff_date = datetime.now() - timedelta(days=days)
        
        for date_folder in self.archive_dir.iterdir():
            if date_folder.is_dir():
                try:
                    folder_date = datetime.strptime(date_folder.name, "%Y-%m-%d")
                    if folder_date < cutoff_date:
                        # Usuń cały folder
                        import shutil
                        shutil.rmtree(date_folder)
                        logger.info(f"Usunięto stare archiwum: {date_folder.name}")
                except ValueError:
                    # Nieprawidłowa nazwa folderu
                    pass
    
    def load_results(self, date: str, simulation_id: str) -> Optional[Dict]:
        """Wczytaj zarchiwizowane wyniki"""
        date_folder = self.archive_dir / date
        
        if not date_folder.exists():
            return None
        
        # Szukaj pliku
        for file_path in date_folder.glob(f"{simulation_id}*"):
            if file_path.suffix == '.joblib':
                return joblib.load(file_path)
            elif file_path.suffix == '.json':
                with open(file_path, 'r') as f:
                    return json.load(f)
        
        return None
    
    def get_archive_summary(self) -> Dict:
        """Podsumowanie archiwum"""
        total_files = 0
        total_size_mb = 0
        dates = []
        
        for date_folder in self.archive_dir.iterdir():
            if date_folder.is_dir():
                dates.append(date_folder.name)
                for file_path in date_folder.iterdir():
                    total_files += 1
                    total_size_mb += file_path.stat().st_size / (1024 * 1024)
        
        return {
            'total_files': total_files,
            'total_size_mb': round(total_size_mb, 2),
            'date_range': {
                'oldest': min(dates) if dates else None,
                'newest': max(dates) if dates else None
            },
            'dates_count': len(dates)
        }


# Test
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Test cache
    cache = CacheManager()
    
    # Test data
    test_data = {
        'array': np.random.randn(100, 100).astype(np.float32),
        'metadata': {'test': True, 'timestamp': datetime.now().isoformat()},
        'nested': {
            'wind': np.random.randn(50, 50).astype(np.float32),
            'temp': np.random.randn(50, 50).astype(np.float32)
        }
    }
    
    # Save
    key = cache.get_key({'test': True, 'timestamp': 'now'})
    cache.save(key, test_data)
    
    # Load
    loaded = cache.load(key)
    if loaded:
        print("✅ Cache działa poprawnie")
        print(f"   Stats: {cache.get_stats()}")
    
    # Test archive
    archive = ResultsArchive()
    archive.save_results("test_sim", test_data)
    print(f"📦 Archive: {archive.get_archive_summary()}")

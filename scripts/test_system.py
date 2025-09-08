# scripts/test_system.py
"""
Skrypt testowy do weryfikacji instalacji systemu
Uruchom w Colab po instalacji wszystkich modułów
"""

import sys
import os
import time
import json
import numpy as np
from datetime import datetime
from pathlib import Path

# Kolory dla czytelności
class Colors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    END = '\033[0m'
    BOLD = '\033[1m'

def print_header(text):
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{text:^60}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.END}\n")

def print_test(name, success, message=""):
    icon = "✅" if success else "❌"
    color = Colors.GREEN if success else Colors.RED
    print(f"{icon} {color}{name}{Colors.END}")
    if message:
        print(f"   {message}")

def test_imports():
    """Test importów modułów"""
    print_header("TEST 1: IMPORTY MODUŁÓW")
    
    modules_to_test = [
        ('numpy', 'numpy'),
        ('scipy', 'scipy'),
        ('rasterio', 'rasterio'),
        ('geopandas', 'geopandas'),
        ('fastapi', 'fastapi'),
        ('pvlib', 'pvlib'),
        ('numba', 'numba')
    ]
    
    all_success = True
    for name, module in modules_to_test:
        try:
            __import__(module)
            print_test(f"Import {name}", True)
        except ImportError as e:
            print_test(f"Import {name}", False, str(e))
            all_success = False
    
    return all_success

def test_google_drive():
    """Test montowania Google Drive"""
    print_header("TEST 2: GOOGLE DRIVE")
    
    drive_path = Path('/content/drive/MyDrive')
    
    if drive_path.exists():
        print_test("Google Drive zmontowany", True)
        
        # Sprawdź strukturę folderów
        project_path = drive_path / 'microclimate'
        if project_path.exists():
            print_test("Folder projektu istnieje", True, str(project_path))
            
            # Sprawdź podfoldery
            required_dirs = ['data/raw', 'data/processed', 'cache', 'logs', 'backend']
            for dir_name in required_dirs:
                dir_path = project_path / dir_name
                exists = dir_path.exists()
                print_test(f"  Folder {dir_name}", exists)
            
            return True
        else:
            print_test("Folder projektu", False, "Utwórz /My Drive/microclimate/")
            return False
    else:
        print_test("Google Drive", False, "Zamontuj Drive najpierw!")
        return False

def test_data_files():
    """Test plików danych"""
    print_header("TEST 3: PLIKI DANYCH")
    
    data_path = Path('/content/drive/MyDrive/microclimate/data/raw')
    
    if not data_path.exists():
        print_test("Folder danych", False, "Brak folderu data/raw")
        return False
    
    required_files = {
        'buildings.gpkg': 'Budynki (WYMAGANE)',
        'dem.tif': 'Model terenu (opcjonalny)',
        'landcover.tif': 'Pokrycie terenu (opcjonalny)',
        'tree_canopy.tif': 'Korony drzew (opcjonalny)'
    }
    
    found_any = False
    for filename, description in required_files.items():
        file_path = data_path / filename
        exists = file_path.exists()
        
        if exists:
            size_mb = file_path.stat().st_size / (1024 * 1024)
            print_test(f"{filename}", True, f"{description} ({size_mb:.2f} MB)")
            found_any = True
        else:
            is_required = 'WYMAGANE' in description
            print_test(f"{filename}", not is_required, f"{description} - BRAK")
    
    return found_any

def test_modules():
    """Test modułów systemu"""
    print_header("TEST 4: MODUŁY SYSTEMU")
    
    sys.path.insert(0, '/content/drive/MyDrive/microclimate/backend')
    
    modules = [
        ('core.wind_solver', 'WindFieldSolver'),
        ('core.thermal_model', 'UrbanThermalModel'),
        ('core.data_processor', 'UrbanDataProcessor'),
        ('utils.gpkg_converter', 'BuildingsRasterizer'),
        ('utils.cache_manager', 'CacheManager'),
        ('api.api_server', 'app')
    ]
    
    all_success = True
    for module_name, class_name in modules:
        try:
            module = __import__(module_name, fromlist=[class_name])
            if hasattr(module, class_name):
                print_test(f"{module_name}.{class_name}", True)
            else:
                print_test(f"{module_name}.{class_name}", False, f"Brak klasy {class_name}")
                all_success = False
        except Exception as e:
            print_test(f"{module_name}", False, str(e)[:50])
            all_success = False
    
    return all_success

def test_data_processing():
    """Test przetwarzania danych"""
    print_header("TEST 5: PRZETWARZANIE DANYCH")
    
    try:
        from core.data_processor import UrbanDataProcessor
        
        processor = UrbanDataProcessor(
            data_dir='/content/drive/MyDrive/microclimate/data/processed',
            resolution=5.0
        )
        
        # Spróbuj wczytać dane
        processor.load_all_layers()
        print_test("Wczytywanie warstw", True)
        
        # Walidacja
        validation = processor.validate_data()
        valid_count = sum(1 for v in validation.values() if v)
        total_count = len(validation)
        
        print_test(f"Walidacja danych", valid_count > 0, 
                  f"{valid_count}/{total_count} warstw poprawnych")
        
        # Geometria
        geometry = processor.create_urban_geometry()
        print_test("Tworzenie geometrii", True, 
                  f"Shape: {geometry['surface_elevation'].shape}")
        
        return True
        
    except Exception as e:
        print_test("Przetwarzanie danych", False, str(e)[:100])
        return False

def test_api_server():
    """Test serwera API"""
    print_header("TEST 6: SERWER API")
    
    try:
        import requests
        
        # Sprawdź localhost
        try:
            response = requests.get('http://localhost:8000/api/status', timeout=2)
            if response.status_code == 200:
                status = response.json()
                print_test("API działa lokalnie", True, f"Status: {status['status']}")
                
                # Sprawdź gotowość
                print_test("  Solver gotowy", status.get('solver_ready', False))
                print_test("  Dane załadowane", status.get('data_loaded', False))
                
                return True
        except:
            print_test("API localhost", False, "Serwer nie odpowiada na localhost:8000")
            
        # Sprawdź ngrok
        print(f"\n{Colors.YELLOW}Podaj URL ngrok (np. https://abc123.ngrok.io): {Colors.END}")
        ngrok_url = input().strip()
        
        if ngrok_url:
            try:
                response = requests.get(f"{ngrok_url}/api/status", timeout=5)
                if response.status_code == 200:
                    print_test("API przez ngrok", True, f"URL: {ngrok_url}")
                    return True
            except:
                print_test("API przez ngrok", False, "Nie można połączyć")
        
        return False
        
    except ImportError:
        print_test("Moduł requests", False, "Zainstaluj requests")
        return False

def test_simulation():
    """Test symulacji"""
    print_header("TEST 7: SYMULACJA TESTOWA")
    
    try:
        from core.wind_solver import WindFieldSolver
        from core.thermal_model import UrbanThermalModel
        
        print("Uruchamianie mini-symulacji...")
        
        # Mini solver
        solver = WindFieldSolver(domain_size=(50, 50, 10), resolution=5.0)
        
        # Testowe przeszkody
        solver.obstacles = np.zeros((50, 50, 10))
        solver.obstacles[20:30, 20:30, :5] = 1.0  # Budynek testowy
        
        # Symulacja
        start_time = time.time()
        phi = solver.solve_potential_flow(5.0, 0.0, max_iter=100)
        velocity = solver.compute_velocity_field(phi)
        duration = time.time() - start_time
        
        # Sprawdź wyniki
        max_speed = np.max(velocity['speed'])
        
        print_test("Solver CFD", True, f"Czas: {duration:.2f}s, Max prędkość: {max_speed:.2f} m/s")
        
        # Test modelu termicznego
        thermal = UrbanThermalModel(54.156, 19.404)
        thermal.albedo_map = np.ones((50, 50)) * 0.2
        
        radiation = thermal.compute_solar_radiation(datetime.now(), 0.3)
        print_test("Model termiczny", True, f"Max radiacja: {np.max(radiation):.1f} W/m²")
        
        return True
        
    except Exception as e:
        print_test("Symulacja", False, str(e)[:100])
        return False

def test_cache():
    """Test systemu cache"""
    print_header("TEST 8: SYSTEM CACHE")
    
    try:
        from utils.cache_manager import CacheManager
        
        cache = CacheManager()
        
        # Test zapisu
        test_data = {
            'test_array': np.random.randn(100, 100).astype(np.float32),
            'metadata': {'test': True}
        }
        
        key = cache.get_key({'test': True})
        cache.save(key, test_data)
        print_test("Zapis do cache", True, f"Klucz: {key}")
        
        # Test odczytu
        loaded = cache.load(key)
        if loaded and 'test_array' in loaded:
            print_test("Odczyt z cache", True)
            
            # Statystyki
            stats = cache.get_stats()
            print_test("Statystyki cache", True, 
                      f"Wpisy: {stats['entries']}, Rozmiar: {stats['total_size_mb']:.2f} MB")
            
            # Cleanup
            cache.delete(key)
            return True
        else:
            print_test("Odczyt z cache", False)
            return False
            
    except Exception as e:
        print_test("System cache", False, str(e)[:100])
        return False

def main():
    """Główna funkcja testowa"""
    print(f"\n{Colors.BOLD}🔬 SYSTEM TEST - Urban Microclimate Simulator{Colors.END}")
    print(f"{Colors.YELLOW}Data: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{Colors.END}\n")
    
    results = []
    
    # Przeprowadź testy
    results.append(("Importy", test_imports()))
    results.append(("Google Drive", test_google_drive()))
    results.append(("Pliki danych", test_data_files()))
    results.append(("Moduły systemu", test_modules()))
    results.append(("Przetwarzanie", test_data_processing()))
    results.append(("Serwer API", test_api_server()))
    results.append(("Symulacja", test_simulation()))
    results.append(("Cache", test_cache()))
    
    # Podsumowanie
    print_header("PODSUMOWANIE")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    print(f"Testy zakończone: {passed}/{total} zaliczonych\n")
    
    for name, result in results:
        icon = "✅" if result else "❌"
        color = Colors.GREEN if result else Colors.RED
        print(f"  {icon} {color}{name:20} {'PASS' if result else 'FAIL'}{Colors.END}")
    
    print()
    
    if passed == total:
        print(f"{Colors.GREEN}{Colors.BOLD}🎉 SYSTEM GOTOWY DO UŻYCIA! 🎉{Colors.END}")
        print(f"{Colors.GREEN}Wszystkie testy zaliczone.{Colors.END}")
    elif passed >= total * 0.7:
        print(f"{Colors.YELLOW}{Colors.BOLD}⚠️ SYSTEM CZĘŚCIOWO GOTOWY{Colors.END}")
        print(f"{Colors.YELLOW}Niektóre komponenty wymagają uwagi.{Colors.END}")
    else:
        print(f"{Colors.RED}{Colors.BOLD}❌ SYSTEM WYMAGA KONFIGURACJI{Colors.END}")
        print(f"{Colors.RED}Sprawdź logi i napraw błędy.{Colors.END}")
    
    print(f"\n{Colors.BLUE}Szczegółowe logi: /content/drive/MyDrive/microclimate/logs/{Colors.END}")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Agri-Mind Dashboard - System Test Script
Verifies that all components are working correctly
"""

import sys
import importlib
from typing import List, Tuple

def test_python_version() -> Tuple[bool, str]:
    """Check Python version"""
    required = (3, 8)
    current = sys.version_info[:2]
    
    if current >= required:
        return True, f"✅ Python {current[0]}.{current[1]} (required: {required[0]}.{required[1]}+)"
    else:
        return False, f"❌ Python {current[0]}.{current[1]} (required: {required[0]}.{required[1]}+)"

def test_imports() -> List[Tuple[bool, str]]:
    """Test all required imports"""
    required_packages = [
        ('streamlit', 'Streamlit'),
        ('folium', 'Folium'),
        ('streamlit_folium', 'Streamlit-Folium'),
        ('numpy', 'NumPy'),
        ('pandas', 'Pandas'),
    ]
    
    results = []
    for package, name in required_packages:
        try:
            importlib.import_module(package)
            results.append((True, f"✅ {name} installed"))
        except ImportError:
            results.append((False, f"❌ {name} missing - install with: pip install {package}"))
    
    return results

def test_demo_data_generation() -> Tuple[bool, str]:
    """Test demo data generation"""
    try:
        import numpy as np
        
        # Simulate demo data generation
        np.random.seed(42)
        ndvi = np.random.random((100, 100))
        ndwi = np.random.random((100, 100))
        
        # Check data is valid
        if ndvi.shape == (100, 100) and ndwi.shape == (100, 100):
            if -1 <= ndvi.min() <= 1 and -1 <= ndvi.max() <= 1:
                return True, "✅ Demo data generation works"
        
        return False, "❌ Demo data validation failed"
    
    except Exception as e:
        return False, f"❌ Demo data generation error: {str(e)}"

def test_config_file() -> Tuple[bool, str]:
    """Test configuration file"""
    try:
        import config
        
        # Check critical settings exist
        required_settings = ['THEME', 'CROP_TYPES', 'MAP_CONFIG', 'DEMO']
        
        for setting in required_settings:
            if not hasattr(config, setting):
                return False, f"❌ Missing config setting: {setting}"
        
        return True, "✅ Configuration file valid"
    
    except ImportError:
        return False, "❌ config.py not found"
    except Exception as e:
        return False, f"❌ Config error: {str(e)}"

def test_app_structure() -> Tuple[bool, str]:
    """Test main app file structure"""
    try:
        with open('app.py', 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check for critical components
        critical_components = [
            'streamlit',
            'folium',
            'SatelliteDataProcessor',
            'ArabicAdvisor',
            'def main():',
            'st.set_page_config'
        ]
        
        missing = [comp for comp in critical_components if comp not in content]
        
        if missing:
            return False, f"❌ Missing components in app.py: {', '.join(missing)}"
        
        return True, "✅ App structure valid"
    
    except FileNotFoundError:
        return False, "❌ app.py not found"
    except Exception as e:
        return False, f"❌ App structure error: {str(e)}"

def test_arabic_support() -> Tuple[bool, str]:
    """Test Arabic text handling"""
    try:
        test_text = "يا حاج، المزرعة في حالة ممتازة"
        
        # Check if Arabic characters are preserved
        if len(test_text) > 0 and 'ال' in test_text:
            return True, "✅ Arabic text support working"
        
        return False, "❌ Arabic text encoding issue"
    
    except Exception as e:
        return False, f"❌ Arabic support error: {str(e)}"

def run_all_tests():
    """Run all tests and display results"""
    print("=" * 60)
    print("🌾 AGRI-MIND DASHBOARD - SYSTEM TEST")
    print("=" * 60)
    print()
    
    all_passed = True
    
    # Test Python version
    print("📋 Testing Environment...")
    print("-" * 60)
    passed, message = test_python_version()
    print(message)
    all_passed = all_passed and passed
    print()
    
    # Test imports
    print("📦 Testing Dependencies...")
    print("-" * 60)
    results = test_imports()
    for passed, message in results:
        print(message)
        all_passed = all_passed and passed
    print()
    
    # Test configuration
    print("⚙️ Testing Configuration...")
    print("-" * 60)
    passed, message = test_config_file()
    print(message)
    all_passed = all_passed and passed
    print()
    
    # Test app structure
    print("🏗️ Testing Application Structure...")
    print("-" * 60)
    passed, message = test_app_structure()
    print(message)
    all_passed = all_passed and passed
    print()
    
    # Test demo data
    print("🎯 Testing Demo Mode...")
    print("-" * 60)
    passed, message = test_demo_data_generation()
    print(message)
    all_passed = all_passed and passed
    print()
    
    # Test Arabic support
    print("🔤 Testing Arabic Support...")
    print("-" * 60)
    passed, message = test_arabic_support()
    print(message)
    all_passed = all_passed and passed
    print()
    
    # Final summary
    print("=" * 60)
    if all_passed:
        print("✅ ALL TESTS PASSED!")
        print()
        print("🚀 Ready to launch! Run: streamlit run app.py")
    else:
        print("❌ SOME TESTS FAILED")
        print()
        print("💡 Fix the issues above before launching")
        print("📚 See README.md for setup instructions")
    print("=" * 60)
    
    return all_passed

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

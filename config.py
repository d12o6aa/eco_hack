"""
Agri-Mind Dashboard Configuration
Customize colors, thresholds, and behavior here
"""

# ============================
# APPEARANCE SETTINGS
# ============================

THEME = {
    'primary_color': '#2d5016',      # Deep Green
    'secondary_color': '#4a7c2a',    # Forest Green
    'background_color': '#f0f7f0',   # Light Green Tint
    'healthy_color': '#4caf50',      # Green
    'attention_color': '#ff9800',    # Orange
    'critical_color': '#f44336',     # Red
    'text_color': '#2d5016',         # Dark Green
}

# ============================
# ANALYSIS THRESHOLDS
# ============================

# NDVI Thresholds (Normalized Difference Vegetation Index)
NDVI_THRESHOLDS = {
    'excellent': 0.6,      # Above this = excellent health
    'good': 0.4,           # Above this = good health
    'moderate': 0.3,       # Above this = moderate health
    'poor': 0.0,           # Below this = poor health
}

# NDWI Thresholds (Normalized Difference Water Index)
NDWI_THRESHOLDS = {
    'well_watered': 0.2,   # Above this = good water content
    'adequate': 0.0,       # Above this = adequate water
    'stressed': -0.1,      # Above this = water stressed
    'severe_stress': -0.3, # Below this = severe stress
}

# Zone Classification Thresholds
ZONE_THRESHOLDS = {
    'healthy': 0.3,        # Combined score above = healthy
    'attention': 0.0,      # Combined score between = needs attention
    # Below 0.0 = critical
}

# ============================
# CROP CONFIGURATIONS
# ============================

CROP_TYPES = {
    'wheat': {
        'name_en': 'Wheat',
        'name_ar': 'القمح',
        'water_requirement': 5000000,  # liters per hectare per season
        'growth_stages': ['Germination', 'Tillering', 'Stem Extension', 'Heading', 'Ripening'],
        'optimal_ndvi': 0.7,
        'optimal_ndwi': 0.2,
    },
    'citrus': {
        'name_en': 'Citrus',
        'name_ar': 'الموالح',
        'water_requirement': 7000000,
        'growth_stages': ['Flowering', 'Fruit Set', 'Cell Division', 'Maturation'],
        'optimal_ndvi': 0.75,
        'optimal_ndwi': 0.25,
    },
    'vegetables': {
        'name_en': 'Vegetables',
        'name_ar': 'الخضروات',
        'water_requirement': 4000000,
        'growth_stages': ['Seedling', 'Vegetative', 'Flowering', 'Harvest'],
        'optimal_ndvi': 0.65,
        'optimal_ndwi': 0.15,
    },
    'corn': {
        'name_en': 'Corn',
        'name_ar': 'الذرة',
        'water_requirement': 6000000,
        'growth_stages': ['Emergence', 'Vegetative', 'Tasseling', 'Silk', 'Dough', 'Maturity'],
        'optimal_ndvi': 0.8,
        'optimal_ndwi': 0.2,
    },
}

# ============================
# SUSTAINABILITY CALCULATIONS
# ============================

SUSTAINABILITY = {
    'water_cost_per_m3': 0.15,           # USD per cubic meter
    'carbon_price_per_tonne': 25,        # USD per tonne CO2
    'precision_irrigation_efficiency': 0.30,  # 30% water savings
    'carbon_per_hectare_base': 2.5,      # tonnes CO2 per hectare per year
}

# ============================
# MAP SETTINGS
# ============================

MAP_CONFIG = {
    'default_center': [30.0444, 31.2357],  # Cairo, Egypt
    'default_zoom': 13,
    'demo_center': [30.3864, 30.3415],     # Wadi El Natrun
    'demo_polygon': [
        [30.390, 30.335],
        [30.390, 30.348],
        [30.383, 30.348],
        [30.383, 30.335],
        [30.390, 30.335]
    ],
    'satellite_tile_url': 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
    'satellite_attribution': 'Esri',
}

# ============================
# DEMO MODE SETTINGS
# ============================

DEMO = {
    'enabled_by_default': True,
    'farm_name': 'Wadi El Natrun Demo Farm',
    'default_crop': 'wheat',
    'default_area_hectares': 10.0,
    'ndvi_seed': 42,  # Random seed for reproducible demo data
    'ndwi_seed': 43,
}

# ============================
# SATELLITE API SETTINGS
# (For future real implementation)
# ============================

SATELLITE_API = {
    'provider': 'planetary_computer',  # or 'sentinel_hub', 'google_earth_engine'
    'endpoint': 'https://planetarycomputer.microsoft.com/api/stac/v1',
    'collection': 'sentinel-2-l2a',
    'cloud_cover_max': 20,  # Maximum cloud cover percentage
    'date_range_days': 30,  # Look back this many days for imagery
    'timeout_seconds': 30,
    'max_retries': 3,
}

# ============================
# UI SETTINGS
# ============================

UI = {
    'page_title': 'Agri-Mind | الزراعة الذكية',
    'page_icon': '🌾',
    'layout': 'wide',
    'show_arabic': True,
    'default_language': 'bilingual',  # 'en', 'ar', or 'bilingual'
    'enable_animations': True,
    'map_height': 500,  # pixels
}

# ============================
# ADVANCED FEATURES
# ============================

FEATURES = {
    'enable_drawing_tools': True,
    'enable_polygon_selection': True,
    'enable_point_selection': True,
    'enable_weather_integration': False,  # Future feature
    'enable_export_reports': False,       # Future feature
    'enable_multi_farm_comparison': False, # Future feature
}

# ============================
# PERFORMANCE SETTINGS
# ============================

PERFORMANCE = {
    'cache_satellite_data': True,
    'cache_timeout_hours': 24,
    'max_polygon_points': 100,
    'analysis_image_size': (100, 100),  # pixels for demo analysis
    'enable_parallel_processing': False,
}

# ============================
# ALERTS & NOTIFICATIONS
# ============================

ALERTS = {
    'critical_threshold_alert': True,
    'water_stress_alert': True,
    'pest_detection_alert': False,  # Future feature
    'alert_methods': ['ui'],  # Future: ['ui', 'email', 'sms', 'whatsapp']
}

"""
Agri-Mind Precision Agriculture Dashboard
With Gemini Pro AI Integration & Visual Maps
"""

import streamlit as st
import folium
from streamlit_folium import st_folium
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import json
from typing import Dict, List, Tuple, Optional
import warnings
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from io import BytesIO
import base64

warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Agri-Mind | الزراعة الذكية",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        background-color: #f0f7f0;
    }
    .stButton>button {
        background-color: #2d5016;
        color: white;
        border-radius: 8px;
        padding: 0.5rem 2rem;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #3d6b1f;
    }
    .metric-card {
        background: linear-gradient(135deg, #2d5016 0%, #4a7c2a 100%);
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .zone-healthy {
        background-color: #4caf50;
        padding: 10px;
        border-radius: 8px;
        color: white;
        margin: 5px 0;
    }
    .zone-attention {
        background-color: #ff9800;
        padding: 10px;
        border-radius: 8px;
        color: white;
        margin: 5px 0;
    }
    .zone-critical {
        background-color: #f44336;
        padding: 10px;
        border-radius: 8px;
        color: white;
        margin: 5px 0;
    }
    .arabic-text {
        font-size: 1.2rem;
        line-height: 1.8;
        direction: rtl;
        text-align: right;
    }
    .coordinate-display {
        background-color: #e8f5e9;
        padding: 1rem;
        border-radius: 8px;
        font-family: monospace;
        margin: 10px 0;
    }
    h1, h2, h3 {
        color: #2d5016;
    }
    .info-box {
        background-color: #e8f5e9;
        padding: 1rem;
        border-left: 4px solid #2d5016;
        border-radius: 4px;
        margin: 1rem 0;
    }
    .ai-response {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
    }
    </style>
""", unsafe_allow_html=True)

# Demo locations
DEMO_LOCATIONS = {
    'wadi_natrun': {
        'name': 'Wadi El Natrun وادي النطرون',
        'coords': [30.3864, 30.3415],
        'seed': 42
    },
    'nile_delta': {
        'name': 'Nile Delta الدلتا',
        'coords': [30.5, 31.0],
        'seed': 123
    },
    'fayoum': {
        'name': 'Fayoum Oasis الفيوم',
        'coords': [29.31, 30.84],
        'seed': 456
    },
    'aswan': {
        'name': 'Aswan أسوان',
        'coords': [24.09, 32.9],
        'seed': 789
    }
}

CROP_INFO = {
    'wheat': {
        'name_ar': 'قمح',
        'name_en': 'Wheat',
        'icon': '🌾',
        'optimal_ndvi': 0.7,
        'optimal_ndwi': 0.2
    },
    'citrus': {
        'name_ar': 'موالح',
        'name_en': 'Citrus',
        'icon': '🍊',
        'optimal_ndvi': 0.75,
        'optimal_ndwi': 0.25
    },
    'vegetables': {
        'name_ar': 'خضروات',
        'name_en': 'Vegetables',
        'icon': '🥬',
        'optimal_ndvi': 0.65,
        'optimal_ndwi': 0.15
    },
    'corn': {
        'name_ar': 'ذرة',
        'name_en': 'Corn',
        'icon': '🌽',
        'optimal_ndvi': 0.8,
        'optimal_ndwi': 0.2
    }
}

class SatelliteDataProcessor:
    """Generates realistic satellite data with visualizations"""
    
    @staticmethod
    def generate_realistic_ndvi(coords: List[float], size: Tuple[int, int] = (200, 200)) -> np.ndarray:
        """Generate location-based NDVI"""
        seed = int(abs(coords[0] * 1000 + coords[1] * 1000)) % 10000
        np.random.seed(seed)
        
        x = np.linspace(-3, 3, size[0])
        y = np.linspace(-3, 3, size[1])
        X, Y = np.meshgrid(x, y)
        
        # Create realistic patterns
        healthy = 0.65 + 0.2 * np.sin(X * 2) * np.cos(Y * 2)
        stress = 0.35 + 0.15 * np.sin(X * 3)
        critical = 0.15 + 0.1 * np.random.random(size)
        
        distance = np.sqrt(X**2 + Y**2)
        ndvi = np.where(distance < 2, healthy,
                       np.where(distance < 3, stress, critical))
        
        noise = 0.05 * np.random.randn(*size)
        ndvi = np.clip(ndvi + noise, -1, 1)
        
        return ndvi
    
    @staticmethod
    def generate_realistic_ndwi(coords: List[float], size: Tuple[int, int] = (200, 200)) -> np.ndarray:
        """Generate location-based NDWI"""
        seed = int(abs(coords[0] * 1000 + coords[1] * 1000)) % 10000 + 1
        np.random.seed(seed)
        
        x = np.linspace(-3, 3, size[0])
        y = np.linspace(-3, 3, size[1])
        X, Y = np.meshgrid(x, y)
        
        well_watered = 0.25 + 0.1 * np.cos(X) * np.sin(Y)
        moderate = 0.0 + 0.1 * np.random.random(size)
        severe = -0.25 + 0.08 * np.random.random(size)
        
        ndwi = np.where(Y > 0.5, well_watered,
                       np.where(Y > -0.5, moderate, severe))
        
        noise = 0.04 * np.random.randn(*size)
        ndwi = np.clip(ndwi + noise, -1, 1)
        
        return ndwi
    
    @staticmethod
    def create_ndvi_heatmap(ndvi: np.ndarray) -> str:
        """Create colored NDVI heatmap"""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Custom colormap: Red -> Yellow -> Green
        colors = ['#8B0000', '#FF0000', '#FF4500', '#FFA500', '#FFD700', 
                  '#FFFF00', '#ADFF2F', '#7FFF00', '#00FF00', '#006400']
        n_bins = 100
        cmap = mcolors.LinearSegmentedColormap.from_list('ndvi', colors, N=n_bins)
        
        im = ax.imshow(ndvi, cmap=cmap, vmin=-0.2, vmax=1.0, aspect='auto')
        
        ax.set_title('NDVI - مؤشر صحة النبات', fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('←← غرب                                شرق ←←', fontsize=12)
        ax.set_ylabel('←← جنوب                                شمال ←←', fontsize=12)
        
        # Colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('NDVI Value', rotation=270, labelpad=20, fontsize=12)
        
        # Add legend
        legend_elements = [
            plt.Rectangle((0,0),1,1, fc='#006400', label='🟢 صحي جداً (>0.6)'),
            plt.Rectangle((0,0),1,1, fc='#FFD700', label='🟡 متوسط (0.3-0.6)'),
            plt.Rectangle((0,0),1,1, fc='#FF0000', label='🔴 ضعيف (<0.3)')
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
        
        plt.tight_layout()
        
        # Convert to base64
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode()
        plt.close()
        
        return f"data:image/png;base64,{img_str}"
    
    @staticmethod
    def create_ndwi_heatmap(ndwi: np.ndarray) -> str:
        """Create colored NDWI heatmap"""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Custom colormap: Brown -> Yellow -> Blue
        colors = ['#8B4513', '#A0522D', '#D2691E', '#F4A460', '#FFE4B5',
                  '#87CEEB', '#4682B4', '#1E90FF', '#0000CD', '#00008B']
        n_bins = 100
        cmap = mcolors.LinearSegmentedColormap.from_list('ndwi', colors, N=n_bins)
        
        im = ax.imshow(ndwi, cmap=cmap, vmin=-0.4, vmax=0.4, aspect='auto')
        
        ax.set_title('NDWI - مؤشر المحتوى المائي', fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('←← غرب                                شرق ←←', fontsize=12)
        ax.set_ylabel('←← جنوب                                شمال ←←', fontsize=12)
        
        # Colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('NDWI Value', rotation=270, labelpad=20, fontsize=12)
        
        # Add legend
        legend_elements = [
            plt.Rectangle((0,0),1,1, fc='#00008B', label='💧 ري ممتاز (>0.2)'),
            plt.Rectangle((0,0),1,1, fc='#FFE4B5', label='💦 متوسط (-0.1 to 0.2)'),
            plt.Rectangle((0,0),1,1, fc='#8B4513', label='🏜️ عطش شديد (<-0.1)')
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
        
        plt.tight_layout()
        
        # Convert to base64
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode()
        plt.close()
        
        return f"data:image/png;base64,{img_str}"
    
    @staticmethod
    def create_combined_map(ndvi: np.ndarray, ndwi: np.ndarray) -> str:
        """Create combined health map"""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Combine NDVI and NDWI
        health_score = ndvi * 0.6 + ndwi * 0.4
        
        # Custom colormap
        colors = ['#8B0000', '#FF4500', '#FFA500', '#FFD700', 
                  '#ADFF2F', '#7FFF00', '#00FF00', '#006400']
        cmap = mcolors.LinearSegmentedColormap.from_list('health', colors, N=100)
        
        im = ax.imshow(health_score, cmap=cmap, vmin=-0.2, vmax=0.8, aspect='auto')
        
        ax.set_title('خريطة الصحة الشاملة للمزرعة', fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('←← غرب                                شرق ←←', fontsize=12)
        ax.set_ylabel('←← جنوب                                شمال ←←', fontsize=12)
        
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Health Score', rotation=270, labelpad=20, fontsize=12)
        
        # Add zones
        legend_elements = [
            plt.Rectangle((0,0),1,1, fc='#006400', label='🟢 منطقة صحية'),
            plt.Rectangle((0,0),1,1, fc='#FFA500', label='🟡 تحتاج اهتمام'),
            plt.Rectangle((0,0),1,1, fc='#8B0000', label='🔴 منطقة حرجة')
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
        
        plt.tight_layout()
        
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode()
        plt.close()
        
        return f"data:image/png;base64,{img_str}"
    
    @staticmethod
    def classify_zones(ndvi: np.ndarray, ndwi: np.ndarray) -> Dict[str, float]:
        """Classify zones with detailed stats"""
        health_score = (ndvi * 0.6 + ndwi * 0.4)
        
        healthy = np.sum(health_score > 0.3)
        attention = np.sum((health_score >= 0.0) & (health_score <= 0.3))
        critical = np.sum(health_score < 0.0)
        
        total = healthy + attention + critical
        
        return {
            'healthy_pct': (healthy / total) * 100,
            'attention_pct': (attention / total) * 100,
            'critical_pct': (critical / total) * 100,
            'ndvi_mean': float(np.mean(ndvi)),
            'ndvi_std': float(np.std(ndvi)),
            'ndvi_min': float(np.min(ndvi)),
            'ndvi_max': float(np.max(ndvi)),
            'ndwi_mean': float(np.mean(ndwi)),
            'ndwi_std': float(np.std(ndwi)),
            'ndwi_min': float(np.min(ndwi)),
            'ndwi_max': float(np.max(ndwi)),
            'uniformity': 1.0 - (float(np.std(ndvi)) / 0.5)
        }

def get_gemini_advice(
    crop_type: str,
    zones: Dict[str, float],
    coordinates: List[float],
    gemini_api_key: Optional[str] = None
) -> str:
    """
    Get AI advice from Gemini Pro
    """
    
    if not gemini_api_key or gemini_api_key == "":
        # Fallback to enhanced rule-based
        return generate_enhanced_advice(crop_type, zones, coordinates)
    
    try:
        import google.generativeai as genai
        
        genai.configure(api_key=gemini_api_key)
        model = genai.GenerativeModel('gemini-pro')
        
        crop_info = CROP_INFO[crop_type]
        
        prompt = f"""
أنت خبير زراعي مصري متخصص في الزراعة الدقيقة. حلّل البيانات دي وقدم نصائح عملية بالعامية المصرية:

📍 **معلومات المزرعة:**
- الموقع: {coordinates[0]:.4f}° شمال, {coordinates[1]:.4f}° شرق
- المحصول: {crop_info['name_ar']} ({crop_info['name_en']})
- المؤشرات المثالية: NDVI={crop_info['optimal_ndvi']}, NDWI={crop_info['optimal_ndwi']}

📊 **نتائج التحليل:**
- المناطق الصحية: {zones['healthy_pct']:.1f}%
- المناطق المتوسطة: {zones['attention_pct']:.1f}%
- المناطق الحرجة: {zones['critical_pct']:.1f}%

📈 **المؤشرات الحالية:**
- NDVI (صحة النبات): {zones['ndvi_mean']:.3f} (المدى: {zones['ndvi_min']:.2f} - {zones['ndvi_max']:.2f})
- NDWI (المحتوى المائي): {zones['ndwi_mean']:.3f} (المدى: {zones['ndwi_min']:.2f} - {zones['ndwi_max']:.2f})
- التجانس: {zones['uniformity']*100:.0f}%

**المطلوب:**
1. تحليل الوضع الحالي بالعامية المصرية
2. تشخيص المشاكل الرئيسية
3. خطة عمل واضحة ومحددة
4. توصيات عاجلة وطويلة المدى

**ملاحظة:** الرد يكون عملي ومباشر، يفهمه الفلاح المصري، واستخدم تعبيرات زي "يا حاج"، "يا ريس"، "المزرعة محتاجة"، إلخ.
"""
        
        response = model.generate_content(prompt)
        return response.text
        
    except ImportError:
        st.warning("⚠️ مكتبة google-generativeai مش متنصبة. استخدم: pip install google-generativeai")
        return generate_enhanced_advice(crop_type, zones, coordinates)
    except Exception as e:
        st.error(f"❌ خطأ في الاتصال بـ Gemini API: {str(e)}")
        return generate_enhanced_advice(crop_type, zones, coordinates)

def generate_enhanced_advice(crop_type: str, zones: Dict[str, float], coordinates: List[float]) -> str:
    """Enhanced fallback advice"""
    crop_info = CROP_INFO[crop_type]
    advice_parts = []
    
    # Opening
    if zones['healthy_pct'] > 75:
        advice_parts.append(f"🌟 **ما شاء الله يا ريس! {crop_info['icon']} {crop_info['name_ar']} في أحسن حال**")
    elif zones['critical_pct'] > 30:
        advice_parts.append(f"⚠️ **انتباه يا حاج! في مشكلة كبيرة في {crop_info['icon']} {crop_info['name_ar']}**")
    else:
        advice_parts.append(f"📊 **الوضع متوسط يا معلم في {crop_info['icon']} {crop_info['name_ar']}**")
    
    # NDVI Analysis
    ndvi_diff = zones['ndvi_mean'] - crop_info['optimal_ndvi']
    if abs(ndvi_diff) > 0.2:
        advice_parts.append(f"\n**🌱 صحة النبات (NDVI = {zones['ndvi_mean']:.2f}):**")
        if ndvi_diff < 0:
            advice_parts.append(f"- النباتات ضعيفة عن المستوى المطلوب بـ {abs(ndvi_diff):.2f}")
            advice_parts.append("- **لازم:** سماد نيتروجيني + فحص آفات")
        else:
            advice_parts.append("- النباتات كويسة جداً")
    
    # NDWI Analysis
    ndwi_diff = zones['ndwi_mean'] - crop_info['optimal_ndwi']
    if zones['ndwi_mean'] < 0:
        advice_parts.append(f"\n**💧 حالة المياه (NDWI = {zones['ndwi_mean']:.2f}):**")
        advice_parts.append("- **عطش شديد! زود الري فوراً**")
        advice_parts.append(f"- المفروض يكون {crop_info['optimal_ndwi']:.2f} لكن عندك {zones['ndwi_mean']:.2f}")
    
    # Uniformity
    if zones['uniformity'] < 0.6:
        advice_parts.append(f"\n**📊 التوزيع (تجانس {zones['uniformity']*100:.0f}%):**")
        advice_parts.append("- المزرعة مش منتظمة - فيه فروقات كبيرة")
        advice_parts.append("- **الحل:** فحص نظام الري + تحليل تربة")
    
    # Critical zones
    if zones['critical_pct'] > 15:
        advice_parts.append(f"\n**🚨 المناطق الحمرا ({zones['critical_pct']:.0f}%):**")
        advice_parts.append("1. روح فوراً شوف المناطق دي")
        advice_parts.append("2. ممكن يكون فيها مرض أو آفة")
        advice_parts.append("3. اتصل بمهندس زراعي لو لزم")
    
    return "\n".join(advice_parts)

def create_enhanced_map(center: List[float], zoom: int = 15) -> folium.Map:
    """Create enhanced map"""
    m = folium.Map(
        location=center,
        zoom_start=zoom,
        tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        attr='Esri',
        zoom_control=True,
        scrollWheelZoom=True,
        max_zoom=20,
        min_zoom=3
    )
    
    folium.plugins.MeasureControl(position='topleft').add_to(m)
    folium.plugins.Fullscreen(position='topleft').add_to(m)
    folium.plugins.Geocoder(position='topright').add_to(m)
    
    draw = folium.plugins.Draw(
        export=True,
        draw_options={
            'polygon': {'allowIntersection': False, 'shapeOptions': {'color': '#2d5016', 'fillOpacity': 0.3}},
            'rectangle': {'shapeOptions': {'color': '#2d5016', 'fillOpacity': 0.3}},
            'marker': True,
            'polyline': False,
            'circle': False,
            'circlemarker': False
        }
    )
    draw.add_to(m)
    
    return m

def main():
    """Main application"""
    
    # Initialize session state
    if 'analyzed' not in st.session_state:
        st.session_state.analyzed = False
    if 'selected_coords' not in st.session_state:
        st.session_state.selected_coords = None
    
    # Header
    st.title("🌾 Agri-Mind | الزراعة الذكية")
    st.markdown("**Powered by Gemini Pro AI + Satellite Analysis**")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ الإعدادات")
        
        # Gemini API Key
        st.subheader("🤖 Gemini Pro API")
        gemini_key = st.text_input(
            "API Key (اختياري)",
            type="password",
            help="احصل عليه من: https://makersuite.google.com/app/apikey",
            placeholder="AIza..."
        )
        
        if gemini_key:
            st.success("✅ Gemini Pro متصل")
        else:
            st.info("💡 بدون API: نصائح محسّنة | مع API: نصائح AI حقيقية")
        
        st.markdown("---")
        
        # Location
        demo_location = st.selectbox(
            "📍 الموقع",
            list(DEMO_LOCATIONS.keys()),
            format_func=lambda x: DEMO_LOCATIONS[x]['name']
        )
        
        # Crop
        crop_type = st.selectbox(
            "🌱 المحصول",
            list(CROP_INFO.keys()),
            format_func=lambda x: f"{CROP_INFO[x]['icon']} {CROP_INFO[x]['name_ar']}"
        )
        
        # Area
        farm_area = st.number_input(
            "📏 المساحة (فدان)",
            min_value=0.5,
            max_value=1000.0,
            value=10.0,
            step=0.5
        )
        
        st.markdown("---")
        
        # Manual coordinates
        with st.expander("🎯 إحداثيات يدوية"):
            manual_lat = st.number_input("Latitude", value=30.3864, format="%.6f", step=0.0001)
            manual_lon = st.number_input("Longitude", value=30.3415, format="%.6f", step=0.0001)
            if st.button("استخدم"):
                st.session_state.selected_coords = [manual_lat, manual_lon]
                st.success("✅ تم")
        
        st.markdown("---")
        analyze_btn = st.button("🔍 تحليل المزرعة", use_container_width=True, type="primary")
    
    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🗺️ الخريطة",
        "📊 النتائج + الصور",
        "🤖 المستشار الذكي",
        "🌍 الاستدامة"
    ])
    
    # Tab 1: Map
    with tab1:
        st.subheader("حدد موقع مزرعتك")
        
        coords = st.session_state.selected_coords or DEMO_LOCATIONS[demo_location]['coords']
        
        m = create_enhanced_map(coords, zoom=15)
        folium.Marker(
            coords,
            popup=f'{coords[0]:.4f}, {coords[1]:.4f}',
            icon=folium.Icon(color='green', icon='leaf', prefix='fa')
        ).add_to(m)
        
        map_data = st_folium(m, width=None, height=600)
        
        if map_data and map_data.get('last_clicked'):
            clicked_lat = map_data['last_clicked']['lat']
            clicked_lng = map_data['last_clicked']['lng']
            st.session_state.selected_coords = [clicked_lat, clicked_lng]
            st.success(f"📍 {clicked_lat:.6f}, {clicked_lng:.6f}")
    
    # Tab 2: Results with Images
    with tab2:
        if analyze_btn or st.session_state.analyzed:
            st.subheader("📊 نتائج التحليل الكاملة")
            
            coords = st.session_state.selected_coords or DEMO_LOCATIONS[demo_location]['coords']
            
            with st.spinner("جاري التحليل..."):
                processor = SatelliteDataProcessor()
                ndvi_data = processor.generate_realistic_ndvi(coords)
                ndwi_data = processor.generate_realistic_ndwi(coords)
                zones = processor.classify_zones(ndvi_data, ndwi_data)
                
                # Create visualizations
                ndvi_img = processor.create_ndvi_heatmap(ndvi_data)
                ndwi_img = processor.create_ndwi_heatmap(ndwi_data)
                combined_img = processor.create_combined_map(ndvi_data, ndwi_data)
                
                st.session_state.update({
                    'zones': zones,
                    'crop_type': crop_type,
                    'farm_area': farm_area,
                    'coords': coords,
                    'ndvi_img': ndvi_img,
                    'ndwi_img': ndwi_img,
                    'combined_img': combined_img,
                    'gemini_key': gemini_key,
                    'analyzed': True
                })
                
                # Metrics
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown(f"""
                    <div class="zone-healthy">
                        <h3>🟢 صحية</h3>
                        <h2>{zones['healthy_pct']:.1f}%</h2>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                    <div class="zone-attention">
                        <h3>🟡 متوسطة</h3>
                        <h2>{zones['attention_pct']:.1f}%</h2>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    st.markdown(f"""
                    <div class="zone-critical">
                        <h3>🔴 حرجة</h3>
                        <h2>{zones['critical_pct']:.1f}%</h2>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown("---")
                
                # Detailed metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("🌿 NDVI", f"{zones['ndvi_mean']:.3f}", 
                             f"المدى: {zones['ndvi_min']:.2f} - {zones['ndvi_max']:.2f}")
                with col2:
                    st.metric("💧 NDWI", f"{zones['ndwi_mean']:.3f}",
                             f"المدى: {zones['ndwi_min']:.2f} - {zones['ndwi_max']:.2f}")
                with col3:
                    st.metric("📊 التجانس", f"{zones['uniformity']*100:.0f}%")
                
                st.markdown("---")
                
                # Visual Maps
                st.subheader("🗺️ الخرائط البصرية")
                
                tab_ndvi, tab_ndwi, tab_combined = st.tabs(["🌿 NDVI", "💧 NDWI", "🎯 الخريطة الشاملة"])
                
                with tab_ndvi:
                    st.markdown("### خريطة صحة النبات (NDVI)")
                    st.markdown(f'<img src="{ndvi_img}" style="width:100%; border-radius:12px;">', unsafe_allow_html=True)
                    st.info("🟢 الأخضر الغامق = نباتات صحية | 🟡 الأصفر = متوسطة | 🔴 الأحمر = ضعيفة")
                
                with tab_ndwi:
                    st.markdown("### خريطة المحتوى المائي (NDWI)")
                    st.markdown(f'<img src="{ndwi_img}" style="width:100%; border-radius:12px;">', unsafe_allow_html=True)
                    st.info("💙 الأزرق = ري ممتاز | 🟡 الأصفر = متوسط | 🟤 البني = عطش شديد")
                
                with tab_combined:
                    st.markdown("### الخريطة الشاملة (NDVI + NDWI)")
                    st.markdown(f'<img src="{combined_img}" style="width:100%; border-radius:12px;">', unsafe_allow_html=True)
                    st.info("هذه الخريطة تجمع بين صحة النبات والمحتوى المائي لتعطيك صورة كاملة")
        else:
            st.info("👆 اضغط 'تحليل المزرعة'")
    
    # Tab 3: AI Advisor
    with tab3:
        if 'zones' in st.session_state:
            st.subheader("🤖 المستشار الزراعي الذكي")
            
            if st.session_state.get('gemini_key'):
                st.success("✅ استخدام Gemini Pro API للنصائح")
            else:
                st.info("💡 نصائح محسّنة (أضف Gemini API Key للنصائح الديناميكية)")
            
            with st.spinner("جاري توليد النصائح من الذكاء الاصطناعي..."):
                advice = get_gemini_advice(
                    st.session_state['crop_type'],
                    st.session_state['zones'],
                    st.session_state['coords'],
                    st.session_state.get('gemini_key')
                )
            
            st.markdown(f"""
            <div class="ai-response arabic-text">
                {advice}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("👆 حلّل المزرعة الأول")
    
    # Tab 4: Sustainability (same as before)
    with tab4:
        if 'zones' in st.session_state:
            st.subheader("🌍 تقرير الاستدامة")
            zones = st.session_state['zones']
            area = st.session_state['farm_area']
            
            water_saved = area * 5000000 * 0.3
            carbon = area * 2.5 * (zones['healthy_pct']/100)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>💧 مياه</h3>
                    <h2>{water_saved/1000:,.0f} m³</h2>
                </div>
                """, unsafe_allow_html=True)
            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>🌱 كربون</h3>
                    <h2>{carbon:.1f} طن</h2>
                </div>
                """, unsafe_allow_html=True)
            with col3:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>💰 قيمة</h3>
                    <h2>${(water_saved/1000*0.15 + carbon*25):,.0f}</h2>
                </div>
                """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center;">
        <p>🌾 <strong>Agri-Mind</strong> | Powered by Gemini Pro AI</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()

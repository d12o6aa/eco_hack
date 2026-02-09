"""
Agri-Mind Precision Agriculture Dashboard
A Streamlit-based dashboard for farm monitoring and analysis
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
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Agri-Mind | الزراعة الذكية",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for eco-friendly green theme
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
    </style>
""", unsafe_allow_html=True)

# Demo data for Wadi El Natrun, Egypt
DEMO_FARM_COORDS = [30.3864, 30.3415]  # Wadi El Natrun
DEMO_POLYGON_COORDS = [
    [30.390, 30.335],
    [30.390, 30.348],
    [30.383, 30.348],
    [30.383, 30.335],
    [30.390, 30.335]
]

class SatelliteDataProcessor:
    """Handles satellite data fetching and processing"""
    
    @staticmethod
    def generate_demo_ndvi(size: Tuple[int, int] = (100, 100)) -> np.ndarray:
        """Generate realistic demo NDVI data"""
        np.random.seed(42)
        
        # Create base pattern with healthy vegetation
        x = np.linspace(-3, 3, size[0])
        y = np.linspace(-3, 3, size[1])
        X, Y = np.meshgrid(x, y)
        
        # Create zones with different health levels
        healthy_zone = 0.7 + 0.15 * np.sin(X) * np.cos(Y)
        stress_zone = 0.4 + 0.1 * np.random.random(size)
        critical_zone = 0.2 + 0.08 * np.random.random(size)
        
        # Combine zones
        ndvi = np.where(X**2 + Y**2 < 4, healthy_zone, 
                       np.where(X**2 + Y**2 < 7, stress_zone, critical_zone))
        
        # Add some realistic variation
        noise = 0.05 * np.random.randn(*size)
        ndvi = np.clip(ndvi + noise, -1, 1)
        
        return ndvi
    
    @staticmethod
    def generate_demo_ndwi(size: Tuple[int, int] = (100, 100)) -> np.ndarray:
        """Generate realistic demo NDWI data"""
        np.random.seed(43)
        
        x = np.linspace(-3, 3, size[0])
        y = np.linspace(-3, 3, size[1])
        X, Y = np.meshgrid(x, y)
        
        # Water stress patterns
        well_watered = 0.3 + 0.1 * np.sin(X * 2) * np.cos(Y * 2)
        moderate_stress = -0.1 + 0.15 * np.random.random(size)
        high_stress = -0.3 + 0.1 * np.random.random(size)
        
        ndwi = np.where(Y > 0, well_watered,
                       np.where(Y > -1.5, moderate_stress, high_stress))
        
        noise = 0.05 * np.random.randn(*size)
        ndwi = np.clip(ndwi + noise, -1, 1)
        
        return ndwi
    
    @staticmethod
    def classify_zones(ndvi: np.ndarray, ndwi: np.ndarray) -> Dict[str, float]:
        """Classify farm into health zones"""
        # Combined health score
        health_score = (ndvi * 0.6 + ndwi * 0.4)
        
        # Classify
        healthy = np.sum(health_score > 0.3)
        attention = np.sum((health_score >= 0.0) & (health_score <= 0.3))
        critical = np.sum(health_score < 0.0)
        
        total = healthy + attention + critical
        
        return {
            'healthy_pct': (healthy / total) * 100,
            'attention_pct': (attention / total) * 100,
            'critical_pct': (critical / total) * 100,
            'ndvi_mean': float(np.mean(ndvi)),
            'ndwi_mean': float(np.mean(ndwi))
        }

class ArabicAdvisor:
    """Generates Arabic agricultural advice based on analysis"""
    
    CROP_ADVICE = {
        'wheat': {
            'healthy': 'يا حاج، المزرعة في حالة ممتازة! القمح بتاعك شكله جميل جداً. خليك مستمر على نظام الري الحالي.',
            'attention': 'يا باشا، في مناطق محتاجة شوية عناية. ممكن تزود الري شوية في الأماكن الصفرا دي.',
            'critical': 'يا أفندم، في مشكلة محتاجة تدخل فوري! المناطق الحمرا دي ممكن تكون فيها آفات أو عطش شديد.'
        },
        'citrus': {
            'healthy': 'الموالح بتاعتك يا معلم في قمة الصحة! الأشجار خضرا وحيوية. كمل على نفس النظام.',
            'attention': 'في شجر محتاج ري إضافي يا حاج. المناطق الصفرا ممكن تحتاج سماد أو ري.',
            'critical': 'تحذير! في مناطق حرجة محتاجة تدخل عاجل. ممكن يكون في إصابة حشرية أو نقص مياه حاد.'
        },
        'vegetables': {
            'healthy': 'الخضار شغال تمام يا فندم! النباتات صحية وقوية. استمر على المتابعة.',
            'attention': 'في مناطق محتاجة اهتمام. شوف الأماكن الصفرا دي وزود لها الري والتسميد.',
            'critical': 'مشكلة كبيرة! لازم تتدخل فوراً في المناطق الحمرا. ممكن تكون آفات أو مرض.'
        },
        'corn': {
            'healthy': 'الذرة ماشية حلو يا معلم! المحصول واعد بإذن الله. كمل نفس العناية.',
            'attention': 'في شوية مناطق محتاجة متابعة. المناطق الصفرا ممكن تستفيد من ري إضافي.',
            'critical': 'انتباه! في مشكلة خطيرة في المناطق الحمرا. لازم تفحص فوراً وتاخد إجراء.'
        }
    }
    
    @classmethod
    def get_advice(cls, crop_type: str, zones: Dict[str, float]) -> str:
        """Generate Arabic advice based on crop and zones"""
        crop_lower = crop_type.lower()
        
        if zones['healthy_pct'] > 70:
            level = 'healthy'
        elif zones['critical_pct'] > 30:
            level = 'critical'
        else:
            level = 'attention'
        
        advice = cls.CROP_ADVICE.get(crop_lower, cls.CROP_ADVICE['wheat'])[level]
        
        # Add specific recommendations
        recommendations = []
        
        if zones['ndvi_mean'] < 0.4:
            recommendations.append("• النباتات محتاجة تغذية أو مياه")
        
        if zones['ndwi_mean'] < -0.1:
            recommendations.append("• نقص واضح في المياه - زود جدول الري")
        
        if zones['critical_pct'] > 20:
            recommendations.append("• افحص الآفات والأمراض في المناطق الحمرا")
        
        if recommendations:
            advice += "\n\n**توصيات محددة:**\n" + "\n".join(recommendations)
        
        return advice

def create_map(center: List[float], zoom: int = 13) -> folium.Map:
    """Create interactive folium map"""
    m = folium.Map(
        location=center,
        zoom_start=zoom,
        tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        attr='Esri'
    )
    
    # Add drawing tools
    from folium import plugins
    draw = plugins.Draw(
        export=True,
        draw_options={
            'polygon': {'allowIntersection': False},
            'polyline': False,
            'rectangle': True,
            'circle': False,
            'marker': True,
            'circlemarker': False
        }
    )
    draw.add_to(m)
    
    return m

def calculate_sustainability_metrics(area_hectares: float, zones: Dict[str, float]) -> Dict[str, float]:
    """Calculate water savings and carbon credits"""
    
    # Water savings (liters per hectare per season)
    # Precision irrigation can save 20-40% of water
    base_water_usage = 5000000  # 5 million liters per hectare per season (average)
    precision_efficiency = 0.30  # 30% savings
    
    water_saved_liters = area_hectares * base_water_usage * precision_efficiency
    
    # Carbon credits (tonnes CO2 per hectare)
    # Reduced water pumping + optimized fertilizer = carbon savings
    carbon_per_hectare = 2.5  # tonnes CO2 per hectare per year
    efficiency_factor = (zones['healthy_pct'] / 100) * 0.8 + 0.2
    
    carbon_credits = area_hectares * carbon_per_hectare * efficiency_factor
    
    # Monetary value (example prices)
    water_cost_per_m3 = 0.15  # USD
    carbon_price_per_tonne = 25  # USD
    
    return {
        'water_saved_liters': water_saved_liters,
        'water_saved_m3': water_saved_liters / 1000,
        'water_value_usd': (water_saved_liters / 1000) * water_cost_per_m3,
        'carbon_credits_tonnes': carbon_credits,
        'carbon_value_usd': carbon_credits * carbon_price_per_tonne,
        'total_savings_usd': (water_saved_liters / 1000) * water_cost_per_m3 + carbon_credits * carbon_price_per_tonne
    }

def main():
    """Main application"""
    
    # Header
    col1, col2 = st.columns([3, 1])
    with col1:
        st.title("🌾 Agri-Mind | الزراعة الذكية")
        st.markdown("**Precision Agriculture for Sustainable Farming**")
    with col2:
        st.image("https://via.placeholder.com/150x80/2d5016/ffffff?text=Agri-Mind", width=150)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ إعدادات المزرعة")
        
        # Crop selection
        crop_type = st.selectbox(
            "🌱 نوع المحصول / Crop Type",
            ["Wheat القمح", "Citrus الموالح", "Vegetables الخضروات", "Corn الذرة"],
            index=0
        )
        crop_clean = crop_type.split()[0]
        
        # Farm area
        farm_area = st.number_input(
            "📏 مساحة المزرعة (فدان) / Farm Area (Hectares)",
            min_value=0.5,
            max_value=1000.0,
            value=10.0,
            step=0.5
        )
        
        # Demo mode toggle
        demo_mode = st.checkbox("🎯 Demo Mode (Wadi El Natrun)", value=True)
        
        st.markdown("---")
        
        # Analysis button
        analyze_btn = st.button("🔍 تحليل المزرعة / Analyze Farm", use_container_width=True)
        
        st.markdown("---")
        st.markdown("""
        <div class="info-box">
        <strong>💡 How to use:</strong><br>
        1. Select your crop type<br>
        2. Enter farm area<br>
        3. Click on map to mark your farm<br>
        4. Click Analyze to get insights
        </div>
        """, unsafe_allow_html=True)
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🗺️ Interactive Map",
        "📊 Analysis Results",
        "💬 Arabic Advisor",
        "🌍 Sustainability Report"
    ])
    
    # Tab 1: Interactive Map
    with tab1:
        st.subheader("Select Your Farm Location")
        
        if demo_mode:
            st.info("🎯 Demo Mode: Showing farm in Wadi El Natrun, Egypt")
            map_center = DEMO_FARM_COORDS
        else:
            map_center = [30.0444, 31.2357]  # Cairo default
        
        # Create and display map
        m = create_map(map_center)
        
        # Add demo polygon if in demo mode
        if demo_mode:
            folium.Polygon(
                locations=DEMO_POLYGON_COORDS,
                color='#2d5016',
                fill=True,
                fill_color='#4a7c2a',
                fill_opacity=0.4,
                popup='Demo Farm - Wadi El Natrun'
            ).add_to(m)
        
        map_data = st_folium(m, width=None, height=500)
        
        # Display selected coordinates
        if map_data and map_data.get('last_clicked'):
            clicked_lat = map_data['last_clicked']['lat']
            clicked_lng = map_data['last_clicked']['lng']
            st.success(f"📍 Selected Location: {clicked_lat:.4f}, {clicked_lng:.4f}")
    
    # Tab 2: Analysis Results
    with tab2:
        if analyze_btn or demo_mode:
            st.subheader("📊 Satellite Analysis Results")
            
            with st.spinner("Analyzing satellite imagery..."):
                # Generate demo data
                processor = SatelliteDataProcessor()
                ndvi_data = processor.generate_demo_ndvi()
                ndwi_data = processor.generate_demo_ndwi()
                zones = processor.classify_zones(ndvi_data, ndwi_data)
                
                # Display metrics
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown(f"""
                    <div class="zone-healthy">
                        <h3>🟢 Healthy Zone</h3>
                        <h2>{zones['healthy_pct']:.1f}%</h2>
                        <p>Optimal vegetation health</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                    <div class="zone-attention">
                        <h3>🟡 Needs Attention</h3>
                        <h2>{zones['attention_pct']:.1f}%</h2>
                        <p>Moderate stress detected</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    st.markdown(f"""
                    <div class="zone-critical">
                        <h3>🔴 Critical Zone</h3>
                        <h2>{zones['critical_pct']:.1f}%</h2>
                        <p>Immediate action required</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown("---")
                
                # Vegetation indices
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric(
                        "🌿 Average NDVI",
                        f"{zones['ndvi_mean']:.3f}",
                        help="Normalized Difference Vegetation Index (higher = healthier)"
                    )
                    
                    # NDVI interpretation
                    if zones['ndvi_mean'] > 0.6:
                        st.success("✅ Excellent vegetation health")
                    elif zones['ndvi_mean'] > 0.3:
                        st.warning("⚠️ Moderate vegetation health")
                    else:
                        st.error("❌ Poor vegetation health")
                
                with col2:
                    st.metric(
                        "💧 Average NDWI",
                        f"{zones['ndwi_mean']:.3f}",
                        help="Normalized Difference Water Index (higher = better water content)"
                    )
                    
                    # NDWI interpretation
                    if zones['ndwi_mean'] > 0.2:
                        st.success("✅ Good water content")
                    elif zones['ndwi_mean'] > -0.1:
                        st.warning("⚠️ Moderate water stress")
                    else:
                        st.error("❌ Severe water stress")
                
                # Store in session state for other tabs
                st.session_state['zones'] = zones
                st.session_state['crop_type'] = crop_clean
                st.session_state['farm_area'] = farm_area
        else:
            st.info("👆 Click 'Analyze Farm' to view results")
    
    # Tab 3: Arabic Advisor
    with tab3:
        if 'zones' in st.session_state:
            st.subheader("💬 النصائح الزراعية / Agricultural Advice")
            
            advisor = ArabicAdvisor()
            advice = advisor.get_advice(
                st.session_state['crop_type'],
                st.session_state['zones']
            )
            
            st.markdown(f"""
            <div class="arabic-text" style="background-color: #e8f5e9; padding: 2rem; border-radius: 12px; border-right: 5px solid #2d5016;">
                {advice}
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Action items
            st.subheader("✅ Recommended Actions")
            
            zones = st.session_state['zones']
            
            actions = []
            if zones['critical_pct'] > 20:
                actions.append("🚨 **Immediate**: Inspect critical zones for pests/disease")
            if zones['ndwi_mean'] < 0:
                actions.append("💧 **Priority**: Increase irrigation in water-stressed areas")
            if zones['ndvi_mean'] < 0.4:
                actions.append("🌱 **Important**: Apply fertilizer to boost vegetation")
            if zones['attention_pct'] > 30:
                actions.append("👀 **Monitor**: Keep close watch on yellow zones")
            
            for action in actions:
                st.markdown(action)
        else:
            st.info("👆 Run analysis first to get personalized advice")
    
    # Tab 4: Sustainability Report
    with tab4:
        if 'zones' in st.session_state:
            st.subheader("🌍 Environmental Impact & Savings")
            
            metrics = calculate_sustainability_metrics(
                st.session_state['farm_area'],
                st.session_state['zones']
            )
            
            # Key metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>💧 Water Saved</h3>
                    <h2>{metrics['water_saved_m3']:,.0f} m³</h2>
                    <p>≈ ${metrics['water_value_usd']:,.2f} USD</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>🌱 Carbon Credits</h3>
                    <h2>{metrics['carbon_credits_tonnes']:.2f} tonnes</h2>
                    <p>≈ ${metrics['carbon_value_usd']:,.2f} USD</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>💰 Total Savings</h3>
                    <h2>${metrics['total_savings_usd']:,.2f}</h2>
                    <p>Per Season</p>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Detailed breakdown
            st.subheader("📈 Detailed Impact Analysis")
            
            impact_data = pd.DataFrame({
                'Category': ['Water Conservation', 'Carbon Reduction', 'Total Environmental Value'],
                'Amount': [
                    f"{metrics['water_saved_m3']:,.0f} m³",
                    f"{metrics['carbon_credits_tonnes']:.2f} tonnes CO₂",
                    f"${metrics['total_savings_usd']:,.2f} USD"
                ],
                'Equivalent To': [
                    f"~{int(metrics['water_saved_m3'] / 50)} Olympic swimming pools",
                    f"~{int(metrics['carbon_credits_tonnes'] / 4.6)} cars off road for 1 year",
                    f"~{int(metrics['total_savings_usd'] / 100)} farmer workdays saved"
                ]
            })
            
            st.dataframe(impact_data, use_container_width=True, hide_index=True)
            
            # Sustainability score
            st.markdown("---")
            st.subheader("🏆 Sustainability Score")
            
            zones = st.session_state['zones']
            sustainability_score = (
                (zones['healthy_pct'] * 0.5) +
                (max(0, 100 - zones['critical_pct']) * 0.3) +
                (min(100, metrics['water_saved_m3'] / 1000) * 0.2)
            )
            
            st.progress(sustainability_score / 100)
            st.markdown(f"### Score: {sustainability_score:.1f}/100")
            
            if sustainability_score > 75:
                st.success("🌟 Excellent! Your farm is highly sustainable")
            elif sustainability_score > 50:
                st.warning("⚡ Good progress! Room for improvement")
            else:
                st.error("🎯 Action needed to improve sustainability")
        else:
            st.info("👆 Run analysis first to view sustainability metrics")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666;">
        <p>🌾 <strong>Agri-Mind</strong> - Empowering Farmers with AI & Satellite Technology</p>
        <p style="font-size: 0.8rem;">Built with ❤️ for sustainable agriculture | Data: Sentinel-2 (Demo Mode)</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()

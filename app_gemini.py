"""
Agri-Mind Precision Agriculture Dashboard - Production Version
Fixed Gemini API + Flexible Inputs + Better UX
"""

import streamlit as st
import folium
from streamlit_folium import st_folium
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import warnings
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from io import BytesIO
import base64

warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Agri-Mind | الزراعة الذكية",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main { background-color: #f0f7f0; }
    .stButton>button {
        background-color: #2d5016;
        color: white;
        border-radius: 8px;
        padding: 0.5rem 2rem;
        font-weight: bold;
    }
    .stButton>button:hover { background-color: #3d6b1f; }
    .metric-card {
        background: linear-gradient(135deg, #2d5016 0%, #4a7c2a 100%);
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .zone-healthy { background-color: #4caf50; padding: 10px; border-radius: 8px; color: white; margin: 5px 0; }
    .zone-attention { background-color: #ff9800; padding: 10px; border-radius: 8px; color: white; margin: 5px 0; }
    .zone-critical { background-color: #f44336; padding: 10px; border-radius: 8px; color: white; margin: 5px 0; }
    .arabic-text { font-size: 1.2rem; line-height: 1.8; direction: rtl; text-align: right; }
    .ai-response {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
    }
    h1, h2, h3 { color: #2d5016; }
    </style>
""", unsafe_allow_html=True)

class SatelliteDataProcessor:
    """Generates realistic satellite data"""
    
    @staticmethod
    def generate_realistic_ndvi(coords: List[float], size: Tuple[int, int] = (200, 200)) -> np.ndarray:
        seed = int(abs(coords[0] * 1000 + coords[1] * 1000)) % 10000
        np.random.seed(seed)
        
        x = np.linspace(-3, 3, size[0])
        y = np.linspace(-3, 3, size[1])
        X, Y = np.meshgrid(x, y)
        
        healthy = 0.65 + 0.2 * np.sin(X * 2) * np.cos(Y * 2)
        stress = 0.35 + 0.15 * np.sin(X * 3)
        critical = 0.15 + 0.1 * np.random.random(size)
        
        distance = np.sqrt(X**2 + Y**2)
        ndvi = np.where(distance < 2, healthy,
                       np.where(distance < 3, stress, critical))
        
        noise = 0.05 * np.random.randn(*size)
        return np.clip(ndvi + noise, -1, 1)
    
    @staticmethod
    def generate_realistic_ndwi(coords: List[float], size: Tuple[int, int] = (200, 200)) -> np.ndarray:
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
        return np.clip(ndwi + noise, -1, 1)
    
    @staticmethod
    def create_heatmap(data: np.ndarray, title: str, cmap_colors: List[str], legend_items: List[Tuple[str, str]]) -> str:
        """Create colored heatmap"""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        cmap = mcolors.LinearSegmentedColormap.from_list('custom', cmap_colors, N=100)
        im = ax.imshow(data, cmap=cmap, aspect='auto')
        
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('←← غرب                                شرق ←←', fontsize=12)
        ax.set_ylabel('←← جنوب                                شمال ←←', fontsize=12)
        
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        legend_elements = [plt.Rectangle((0,0),1,1, fc=color, label=label) 
                          for color, label in legend_items]
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

def get_gemini_advice(crop_name: str, zones: Dict[str, float], coords: List[float], api_key: Optional[str]) -> str:
    """Get AI advice from Gemini"""
    
    if not api_key:
        return generate_fallback_advice(crop_name, zones, coords)
    
    try:
        import google.generativeai as genai
        
        # Configure with API key
        genai.configure(api_key=api_key)
        
        # Use gemini-1.5-flash (the correct model name)
        model = genai.GenerativeModel('gemini-2.5-flash')
        
        prompt = f"""
أنت خبير زراعي مصري متخصص في الزراعة الدقيقة. حلّل البيانات التالية وقدم نصائح عملية بالعامية المصرية:

📍 **معلومات المزرعة:**
- الموقع: {coords[0]:.4f}° شمال, {coords[1]:.4f}° شرق
- المحصول: {crop_name}

📊 **نتائج التحليل:**
- المناطق الصحية: {zones['healthy_pct']:.1f}%
- المناطق المتوسطة: {zones['attention_pct']:.1f}%
- المناطق الحرجة: {zones['critical_pct']:.1f}%

📈 **المؤشرات:**
- NDVI (صحة النبات): {zones['ndvi_mean']:.3f} (المدى: {zones['ndvi_min']:.2f} - {zones['ndvi_max']:.2f})
- NDWI (المحتوى المائي): {zones['ndwi_mean']:.3f} (المدى: {zones['ndwi_min']:.2f} - {zones['ndwi_max']:.2f})
- التجانس: {zones['uniformity']*100:.0f}%

**المطلوب:**
اكتب نصيحة عملية للمزارع بالعامية المصرية. استخدم:
- تعبيرات زي "يا حاج"، "يا ريس"، "المزرعة محتاجة"
- نصائح عملية ومحددة
- خطوات واضحة
- أولويات (فوري، مهم، متابعة)

الرد يكون مباشر وسهل الفهم للفلاح المصري.
"""
        
        response = model.generate_content(prompt)
        return response.text
        
    except ImportError:
        st.warning("⚠️ مكتبة google-generativeai مش منصّبة. استخدم: `pip install google-generativeai`")
        return generate_fallback_advice(crop_name, zones, coords)
    except Exception as e:
        st.error(f"❌ خطأ في Gemini API: {str(e)}")
        st.info("💡 استخدم النصائح المحسّنة بدلاً من ذلك")
        return generate_fallback_advice(crop_name, zones, coords)

def generate_fallback_advice(crop_name: str, zones: Dict[str, float], coords: List[float]) -> str:
    """Fallback advice without AI"""
    advice = []
    
    # Opening
    if zones['healthy_pct'] > 75:
        advice.append(f"🌟 **ما شاء الله يا ريس! {crop_name} في أحسن حال**\n")
        advice.append(f"المزرعة شغالة تمام - {zones['healthy_pct']:.0f}% من المساحة في حالة ممتازة.")
    elif zones['critical_pct'] > 30:
        advice.append(f"⚠️ **انتباه يا حاج! {crop_name} محتاج تدخل فوري**\n")
        advice.append(f"في {zones['critical_pct']:.0f}% من المزرعة في حالة حرجة - لازم نتصرف بسرعة.")
    else:
        advice.append(f"📊 **الوضع متوسط يا معلم في {crop_name}**\n")
        advice.append(f"المزرعة محتاجة شوية اهتمام عشان نحسّن الإنتاج.")
    
    # NDVI Analysis
    advice.append(f"\n**🌱 صحة النبات (NDVI = {zones['ndvi_mean']:.2f}):**")
    if zones['ndvi_mean'] < 0.3:
        advice.append("- النباتات ضعيفة جداً")
        advice.append("- **لازم فوراً:** سماد نيتروجيني + فحص آفات وأمراض")
        advice.append("- ممكن يكون في نقص عناصر أو إصابة")
    elif zones['ndvi_mean'] < 0.5:
        advice.append("- النباتات في حالة متوسطة")
        advice.append("- زود السماد تدريجياً")
        advice.append("- راقب المناطق الحمرا كل يومين")
    else:
        advice.append("- النباتات صحية وقوية")
        advice.append("- استمر على نفس برنامج التسميد")
    
    # NDWI Analysis
    advice.append(f"\n**💧 حالة المياه (NDWI = {zones['ndwi_mean']:.2f}):**")
    if zones['ndwi_mean'] < -0.1:
        advice.append("- **عطش شديد! زود الري فوراً**")
        advice.append("- النباتات بتعاني من نقص مياه حاد")
        advice.append("- شوف نظام الري لو فيه انسداد أو مشكلة")
    elif zones['ndwi_mean'] < 0.1:
        advice.append("- الري مقبول بس ممكن يتحسّن")
        advice.append("- زود فترات الري في المناطق الجافة")
    else:
        advice.append("- الري ممتاز - المياه كافية")
        advice.append("- حافظ على نفس الجدول")
    
    # Uniformity
    if zones['uniformity'] < 0.6:
        advice.append(f"\n**📊 التوزيع (تجانس {zones['uniformity']*100:.0f}%):**")
        advice.append("- المزرعة مش منتظمة - فيه فروقات كبيرة بين المناطق")
        advice.append("- **الحل:**")
        advice.append("  1. افحص نظام الري - ممكن يكون فيه مناطق مش واصلها مياه كويس")
        advice.append("  2. خد عينات تربة من المناطق المختلفة")
        advice.append("  3. راجع توزيع السماد")
    
    # Critical zones action
    if zones['critical_pct'] > 15:
        advice.append(f"\n**🚨 خطة طوارئ للمناطق الحمرا ({zones['critical_pct']:.0f}%):**")
        advice.append("**اليوم:**")
        advice.append("- روح فوراً افحص المناطق الحمرا بنفسك")
        advice.append("- شوف لو فيه آفات، أمراض، أو مشاكل في الري")
        advice.append("\n**هذا الأسبوع:**")
        advice.append("- خد عينات من النباتات المريضة للتحليل")
        advice.append("- اتصل بمهندس زراعي لو لقيت حاجة مش فاهمها")
        advice.append("- ابدأ علاج فوري حسب المشكلة")
    
    # Location-specific
    lat = coords[0]
    if lat > 30:  # Delta region
        advice.append("\n**💡 نصيحة خاصة بمنطقة الدلتا:**")
        advice.append("- راقب الملوحة في التربة")
        advice.append("- الصرف مهم جداً في المنطقة دي")
    elif lat < 26:  # Upper Egypt
        advice.append("\n**💡 نصيحة خاصة بالصعيد:**")
        advice.append("- الحرارة عالية - زود الري في الصيف")
        advice.append("- اهتم بالتسميد العضوي")
    
    advice.append(f"\n---\n📍 الموقع: {coords[0]:.4f}°N, {coords[1]:.4f}°E")
    
    return "\n".join(advice)

def create_map(center: List[float], zoom: int = 13) -> folium.Map:
    """Create interactive map"""
    m = folium.Map(
        location=center,
        zoom_start=zoom,
        tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        attr='Esri',
        zoom_control=True,
        scrollWheelZoom=True,
        max_zoom=20
    )
    
    folium.plugins.MeasureControl(position='topleft').add_to(m)
    folium.plugins.Fullscreen(position='topleft').add_to(m)
    folium.plugins.Geocoder(position='topright').add_to(m)
    
    draw = folium.plugins.Draw(
        export=True,
        draw_options={
            'polygon': {'allowIntersection': False, 'shapeOptions': {'color': '#2d5016'}},
            'rectangle': {'shapeOptions': {'color': '#2d5016'}},
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
    
    if 'analyzed' not in st.session_state:
        st.session_state.analyzed = False
    if 'coords' not in st.session_state:
        st.session_state.coords = [30.0, 31.0]  # Default Cairo
    
    # Header
    st.title("🌾 Agri-Mind | الزراعة الذكية")
    st.markdown("**Precision Agriculture with AI + Satellite Analysis**")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ الإعدادات")
        
        # Gemini API
        st.subheader("🤖 Gemini AI (اختياري)")
        gemini_key = st.text_input(
            "API Key",
            type="password",
            help="احصل عليه مجاناً من: https://aistudio.google.com/app/apikey",
            placeholder="AIza..."
        )
        
        if gemini_key:
            st.success("✅ Gemini AI متصل")
        else:
            st.info("💡 بدون API: نصائح محسّنة | مع API: AI ديناميكي")
        
        st.markdown("---")
        
        # Coordinates input
        st.subheader("📍 موقع المزرعة")
        
        input_method = st.radio(
            "طريقة الإدخال:",
            ["اضغط على الخريطة", "إحداثيات يدوية"],
            help="اختر كيف تحب تحدد موقع مزرعتك"
        )
        
        if input_method == "إحداثيات يدوية":
            lat = st.number_input(
                "Latitude (خط العرض)",
                value=st.session_state.coords[0],
                min_value=-90.0,
                max_value=90.0,
                format="%.6f",
                step=0.0001,
                help="مثال: 30.3864 (شمال مصر)"
            )
            lon = st.number_input(
                "Longitude (خط الطول)",
                value=st.session_state.coords[1],
                min_value=-180.0,
                max_value=180.0,
                format="%.6f",
                step=0.0001,
                help="مثال: 30.3415 (شرق مصر)"
            )
            st.session_state.coords = [lat, lon]
            st.success(f"✅ الموقع: {lat:.4f}, {lon:.4f}")
        else:
            st.info("👆 اضغط على الخريطة لتحديد الموقع")
        
        st.markdown("---")
        
        # Crop - Open text input
        st.subheader("🌱 المحصول")
        
        crop_suggestions = [
            "قمح / Wheat",
            "ذرة / Corn", 
            "أرز / Rice",
            "قطن / Cotton",
            "قصب السكر / Sugarcane",
            "برسيم / Clover",
            "بطاطس / Potato",
            "طماطم / Tomato",
            "موالح (برتقال/ليمون) / Citrus",
            "عنب / Grapes",
            "مانجو / Mango",
            "نخيل / Palm",
            "فول / Beans",
            "بصل / Onion",
            "ثوم / Garlic"
        ]
        
        crop_input_type = st.radio(
            "اختيار المحصول:",
            ["من القائمة", "كتابة حرة"],
            horizontal=True
        )
        
        if crop_input_type == "من القائمة":
            crop_name = st.selectbox("اختر المحصول:", crop_suggestions)
        else:
            crop_name = st.text_input(
                "اكتب اسم المحصول:",
                placeholder="مثال: بنجر السكر",
                help="اكتب أي محصول تزرعه"
            )
            if not crop_name:
                crop_name = "محصول غير محدد"
        
        st.markdown("---")
        
        # Area
        farm_area = st.number_input(
            "📏 المساحة (فدان/هكتار)",
            min_value=0.1,
            max_value=10000.0,
            value=10.0,
            step=0.5
        )
        
        st.markdown("---")
        analyze_btn = st.button("🔍 تحليل المزرعة", use_container_width=True, type="primary")
        
        st.markdown("---")
        st.caption("""
        **💡 نصائح:**
        - استخدم Gemini API للنصائح الذكية
        - حدد الموقع بدقة للنتائج الأفضل
        - جرّب مواقع مختلفة لمقارنة النتائج
        """)
    
    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🗺️ الخريطة",
        "📊 النتائج + الصور",
        "🤖 المستشار الذكي",
        "🌍 الاستدامة"
    ])
    
    # Tab 1: Map
    with tab1:
        st.subheader("حدد موقع مزرعتك على الخريطة")
        
        m = create_map(st.session_state.coords, zoom=13)
        
        folium.Marker(
            st.session_state.coords,
            popup=f'{st.session_state.coords[0]:.4f}, {st.session_state.coords[1]:.4f}',
            icon=folium.Icon(color='green', icon='leaf', prefix='fa')
        ).add_to(m)
        
        map_data = st_folium(m, width=None, height=600)
        
        if map_data and map_data.get('last_clicked'):
            clicked_lat = map_data['last_clicked']['lat']
            clicked_lng = map_data['last_clicked']['lng']
            st.session_state.coords = [clicked_lat, clicked_lng]
            st.success(f"📍 موقع جديد: {clicked_lat:.6f}, {clicked_lng:.6f}")
    
    # Tab 2: Results
    with tab2:
        if analyze_btn or st.session_state.analyzed:
            st.subheader("📊 نتائج التحليل")
            
            coords = st.session_state.coords
            
            with st.spinner("جاري التحليل..."):
                processor = SatelliteDataProcessor()
                ndvi = processor.generate_realistic_ndvi(coords)
                ndwi = processor.generate_realistic_ndwi(coords)
                zones = processor.classify_zones(ndvi, ndwi)
                
                # Create visualizations
                ndvi_colors = ['#8B0000', '#FF0000', '#FF4500', '#FFA500', '#FFD700', 
                              '#FFFF00', '#ADFF2F', '#7FFF00', '#00FF00', '#006400']
                ndvi_img = processor.create_heatmap(
                    ndvi,
                    'NDVI - مؤشر صحة النبات',
                    ndvi_colors,
                    [('#006400', '🟢 صحي (>0.6)'), ('#FFD700', '🟡 متوسط (0.3-0.6)'), ('#FF0000', '🔴 ضعيف (<0.3)')]
                )
                
                ndwi_colors = ['#8B4513', '#A0522D', '#D2691E', '#F4A460', '#FFE4B5',
                              '#87CEEB', '#4682B4', '#1E90FF', '#0000CD', '#00008B']
                ndwi_img = processor.create_heatmap(
                    ndwi,
                    'NDWI - مؤشر المحتوى المائي',
                    ndwi_colors,
                    [('#00008B', '💧 ممتاز (>0.2)'), ('#FFE4B5', '💦 متوسط'), ('#8B4513', '🏜️ عطش (<-0.1)')]
                )
                
                health = ndvi * 0.6 + ndwi * 0.4
                combined_colors = ['#8B0000', '#FF4500', '#FFA500', '#FFD700', 
                                  '#ADFF2F', '#7FFF00', '#00FF00', '#006400']
                combined_img = processor.create_heatmap(
                    health,
                    'الخريطة الشاملة للمزرعة',
                    combined_colors,
                    [('#006400', '🟢 صحية'), ('#FFA500', '🟡 متوسطة'), ('#8B0000', '🔴 حرجة')]
                )
                
                st.session_state.update({
                    'zones': zones,
                    'crop_name': crop_name,
                    'farm_area': farm_area,
                    'ndvi_img': ndvi_img,
                    'ndwi_img': ndwi_img,
                    'combined_img': combined_img,
                    'gemini_key': gemini_key,
                    'analyzed': True
                })
                
                # Metrics
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown(f'<div class="zone-healthy"><h3>🟢 صحية</h3><h2>{zones["healthy_pct"]:.1f}%</h2></div>', unsafe_allow_html=True)
                with col2:
                    st.markdown(f'<div class="zone-attention"><h3>🟡 متوسطة</h3><h2>{zones["attention_pct"]:.1f}%</h2></div>', unsafe_allow_html=True)
                with col3:
                    st.markdown(f'<div class="zone-critical"><h3>🔴 حرجة</h3><h2>{zones["critical_pct"]:.1f}%</h2></div>', unsafe_allow_html=True)
                
                st.markdown("---")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("🌿 NDVI", f"{zones['ndvi_mean']:.3f}", f"{zones['ndvi_min']:.2f} - {zones['ndvi_max']:.2f}")
                with col2:
                    st.metric("💧 NDWI", f"{zones['ndwi_mean']:.3f}", f"{zones['ndwi_min']:.2f} - {zones['ndwi_max']:.2f}")
                with col3:
                    st.metric("📊 التجانس", f"{zones['uniformity']*100:.0f}%")
                
                st.markdown("---")
                
                # Visual maps
                st.subheader("🗺️ الخرائط البصرية")
                tab_n, tab_w, tab_c = st.tabs(["🌿 NDVI", "💧 NDWI", "🎯 شاملة"])
                
                with tab_n:
                    st.markdown(f'<img src="{ndvi_img}" style="width:100%; border-radius:12px;">', unsafe_allow_html=True)
                with tab_w:
                    st.markdown(f'<img src="{ndwi_img}" style="width:100%; border-radius:12px;">', unsafe_allow_html=True)
                with tab_c:
                    st.markdown(f'<img src="{combined_img}" style="width:100%; border-radius:12px;">', unsafe_allow_html=True)
        else:
            st.info("👆 اضغط 'تحليل المزرعة'")
    
    # Tab 3: AI Advisor
    with tab3:
        if 'zones' in st.session_state:
            st.subheader("🤖 المستشار الزراعي الذكي")
            
            if st.session_state.get('gemini_key'):
                st.success("✅ استخدام Gemini 1.5 Flash")
            else:
                st.info("💡 نصائح محسّنة (أضف API للنصائح الذكية)")
            
            with st.spinner("جاري توليد النصائح..."):
                advice = get_gemini_advice(
                    st.session_state['crop_name'],
                    st.session_state['zones'],
                    st.session_state.coords,
                    st.session_state.get('gemini_key')
                )
            
            st.markdown(f'<div class="ai-response arabic-text">{advice}</div>', unsafe_allow_html=True)
        else:
            st.info("👆 حلّل المزرعة أولاً")
    
    # Tab 4: Sustainability
    with tab4:
        if 'zones' in st.session_state:
            st.subheader("🌍 تقرير الاستدامة")
            
            zones = st.session_state['zones']
            area = st.session_state['farm_area']
            
            water_saved = area * 5000000 * 0.3
            carbon = area * 2.5 * (zones['healthy_pct']/100)
            value = (water_saved/1000*0.15 + carbon*25)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(f'<div class="metric-card"><h3>💧 مياه</h3><h2>{water_saved/1000:,.0f} m³</h2><p>${water_saved/1000*0.15:,.0f}</p></div>', unsafe_allow_html=True)
            with col2:
                st.markdown(f'<div class="metric-card"><h3>🌱 كربون</h3><h2>{carbon:.1f} طن</h2><p>${carbon*25:,.0f}</p></div>', unsafe_allow_html=True)
            with col3:
                st.markdown(f'<div class="metric-card"><h3>💰 قيمة</h3><h2>${value:,.0f}</h2><p>في الموسم</p></div>', unsafe_allow_html=True)
            
            st.markdown("---")
            
            impact_df = pd.DataFrame({
                'المؤشر': ['💧 توفير المياه', '🌿 خفض الكربون', '💵 القيمة'],
                'الكمية': [f"{water_saved/1000:,.0f} m³", f"{carbon:.1f} طن CO₂", f"${value:,.0f}"],
                'يعادل': [
                    f"{int(water_saved/1000/50)} حمام سباحة",
                    f"{int(carbon/4.6)} سيارة متوقفة سنة",
                    f"{int(value/100)} يوم عمل"
                ]
            })
            st.dataframe(impact_df, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    st.markdown('<div style="text-align:center;"><p>🌾 <strong>Agri-Mind</strong> | Powered by Gemini AI</p></div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
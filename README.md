# 🌾 Agri-Mind - Precision Agriculture Dashboard

A sophisticated Streamlit-based dashboard for precision agriculture, combining satellite imagery analysis with AI-powered insights in both English and Arabic (Egyptian dialect).

## 🚀 Features

### 1. **Interactive Mapping System**
- Google Maps-like satellite view interface
- Click-to-select farm location
- Polygon drawing tools for precise area selection
- Pre-loaded demo farm in Wadi El Natrun, Egypt

### 2. **Satellite Data Analysis**
- **NDVI (Normalized Difference Vegetation Index)**: Detects vegetation health and pest issues
- **NDWI (Normalized Difference Water Index)**: Identifies water stress areas
- Automatic classification into 3 zones:
  - 🟢 **Healthy (Green)**: Optimal vegetation health
  - 🟡 **Needs Attention (Yellow)**: Moderate stress detected
  - 🔴 **Critical (Red)**: Immediate action required

### 3. **AI-Powered Arabic Advisor** 
The "Mazar3 Mode" translates technical analysis into practical Egyptian Arabic advice:
- Crop-specific recommendations for:
  - Wheat (القمح)
  - Citrus (الموالح)
  - Vegetables (الخضروات)
  - Corn (الذرة)
- Natural language explanations: "يا حاج، المنطقة دي محتاجة ري فوراً"
- Actionable insights in familiar dialect

### 4. **Sustainability Impact Report**
- **Water Conservation**: Estimated liters/m³ saved per season
- **Carbon Credits**: CO₂ reduction in tonnes
- **Economic Value**: USD savings from precision agriculture
- Sustainability score with gamification

### 5. **Demo Mode** 🎯
Built-in fallback with realistic pre-loaded data for Wadi El Natrun farm to ensure flawless pitch presentations even without live API connection.

## 📦 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup Steps

1. **Clone or download the project**
```bash
cd agri-mind-dashboard
```

2. **Create a virtual environment (recommended)**
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Mac/Linux
python3 -m venv venv
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Run the application**
```bash
streamlit run app.py
```

5. **Access the dashboard**
Open your browser to: `http://localhost:8501`

## 🎯 How to Use

### For the Pitch/Demo:
1. Keep **Demo Mode** enabled (checked by default)
2. The map will show a pre-configured farm in Wadi El Natrun
3. Click **"Analyze Farm"** to see instant results
4. Navigate through tabs to showcase all features

### For Real Usage:
1. Uncheck **Demo Mode** 
2. Click on the map to mark your farm location or draw a polygon
3. Select your crop type from the sidebar
4. Enter your farm area in hectares (فدان)
5. Click **"Analyze Farm"**
6. Review results across all tabs

## 📊 Dashboard Tabs

### 🗺️ Tab 1: Interactive Map
- Satellite base layer (ESRI World Imagery)
- Drawing tools for polygon selection
- Location marker placement
- Coordinate display

### 📊 Tab 2: Analysis Results
- Zone classification percentages
- NDVI & NDWI average values
- Color-coded health metrics
- Visual interpretation guides

### 💬 Tab 3: Arabic Advisor
- Contextual advice in Egyptian dialect
- Crop-specific recommendations
- Prioritized action items
- Easy-to-understand insights

### 🌍 Tab 4: Sustainability Report
- Water savings calculator
- Carbon credit estimations
- Economic impact analysis
- Environmental equivalencies
- Sustainability scoring system

## 🛠️ Technical Architecture

### Core Technologies
- **Frontend**: Streamlit
- **Mapping**: Folium + Streamlit-Folium
- **Data Processing**: NumPy, Pandas
- **Satellite Data**: PySTAC-Client, StackSTAC, Rasterio (for future real API integration)

### Data Flow
```
User Input (Map Click) 
    → Coordinate Extraction
    → Satellite Data Fetch (or Demo Data)
    → NDVI/NDWI Calculation
    → Zone Classification
    → Arabic Advice Generation
    → Sustainability Metrics
    → Dashboard Rendering
```

### Key Components

**SatelliteDataProcessor Class**
- Generates realistic NDVI/NDWI patterns
- Performs zone classification
- Ready for real Sentinel-2 API integration

**ArabicAdvisor Class**
- Crop-specific advice database
- Contextual recommendation engine
- Egyptian dialect translations

**Sustainability Calculator**
- Water conservation estimates
- Carbon credit calculations
- Economic value projections

## 🎨 Design System

**Color Palette**
- Primary: `#2d5016` (Deep Green)
- Secondary: `#4a7c2a` (Forest Green)
- Background: `#f0f7f0` (Light Green Tint)
- Healthy: `#4caf50` (Green)
- Attention: `#ff9800` (Orange)
- Critical: `#f44336` (Red)

**Typography**
- Headers: Bold, Green (`#2d5016`)
- Arabic Text: RTL, 1.2rem, Readable line-height
- Metrics: Large, Bold, White on gradient cards

## 🔮 Future Enhancements

### Real Satellite Integration
Replace demo data with live Sentinel-2 API:
```python
from pystac_client import Client
import stackstac

# Connect to Microsoft Planetary Computer
catalog = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")

# Search for Sentinel-2 imagery
search = catalog.search(
    collections=["sentinel-2-l2a"],
    bbox=[min_lon, min_lat, max_lon, max_lat],
    datetime="2024-01-01/2024-12-31"
)

# Stack and process
stack = stackstac.stack(search.items())
```

### Advanced Features
- [ ] Multi-temporal analysis (season-over-season trends)
- [ ] Weather integration (forecasts + recommendations)
- [ ] Pest detection using deep learning
- [ ] Mobile app version
- [ ] Multi-farm comparison dashboard
- [ ] Automated irrigation system integration
- [ ] SMS/WhatsApp alert system

## 📱 Mobile Responsiveness
The dashboard is optimized for desktop but remains functional on tablets. For best experience, use a screen width of 1280px or higher.

## 🐛 Troubleshooting

**Issue**: Map not loading
- **Solution**: Check internet connection (satellite tiles require internet)

**Issue**: Analysis button not responding
- **Solution**: Ensure a crop type is selected and farm area is entered

**Issue**: Arabic text not displaying correctly
- **Solution**: Ensure your browser supports RTL text rendering

## 💡 Pro Tips for Pitch

1. **Start with Demo Mode ON** - Instant results, no waiting
2. **Emphasize the Arabic advisor** - Key differentiator for Egyptian market
3. **Showcase sustainability metrics** - ESG appeal for investors
4. **Mention scalability** - "Built for one farm, scales to thousands"
5. **Highlight cost savings** - Concrete USD values resonate with farmers

## 📄 License
This is a demo application for Agri-Mind startup pitch. Commercial use requires proper Sentinel-2 API credentials and licensing.

## 🤝 Support
For questions about the dashboard implementation, refer to the inline code comments or contact the development team.

---

**Built with ❤️ for sustainable agriculture in Egypt and beyond** 🌾🇪🇬

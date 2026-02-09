"""
Real Sentinel-2 Satellite Data Integration Module
This module replaces demo data with actual satellite imagery

SETUP REQUIRED:
1. Install: pip install pystac-client planetary-computer stackstac rasterio
2. Get API key from Microsoft Planetary Computer (free tier available)
3. Update SATELLITE_API settings in config.py
"""

import numpy as np
import rasterio
from typing import Tuple, Optional, List
from datetime import datetime, timedelta

try:
    from pystac_client import Client
    import stackstac
    import planetary_computer as pc
    SATELLITE_AVAILABLE = True
except ImportError:
    SATELLITE_AVAILABLE = False
    print("⚠️ Satellite libraries not installed. Using demo mode only.")


class SentinelDataFetcher:
    """
    Fetches and processes real Sentinel-2 satellite imagery
    """
    
    def __init__(self, api_endpoint: str = "https://planetarycomputer.microsoft.com/api/stac/v1"):
        """
        Initialize the Sentinel-2 data fetcher
        
        Args:
            api_endpoint: STAC API endpoint URL
        """
        if not SATELLITE_AVAILABLE:
            raise ImportError("Satellite libraries not installed. Run: pip install pystac-client planetary-computer stackstac")
        
        self.catalog = Client.open(api_endpoint)
        self.collection = "sentinel-2-l2a"
    
    def search_imagery(
        self,
        bbox: List[float],
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        max_cloud_cover: int = 20
    ) -> List:
        """
        Search for Sentinel-2 imagery
        
        Args:
            bbox: Bounding box [min_lon, min_lat, max_lon, max_lat]
            start_date: Start date for search (defaults to 30 days ago)
            end_date: End date for search (defaults to today)
            max_cloud_cover: Maximum cloud cover percentage (0-100)
        
        Returns:
            List of STAC items
        """
        if end_date is None:
            end_date = datetime.now()
        if start_date is None:
            start_date = end_date - timedelta(days=30)
        
        # Format dates for STAC API
        date_range = f"{start_date.strftime('%Y-%m-%d')}/{end_date.strftime('%Y-%m-%d')}"
        
        # Search for imagery
        search = self.catalog.search(
            collections=[self.collection],
            bbox=bbox,
            datetime=date_range,
            query={"eo:cloud_cover": {"lt": max_cloud_cover}}
        )
        
        items = list(search.items())
        print(f"✅ Found {len(items)} Sentinel-2 scenes")
        
        return items
    
    def get_bbox_from_polygon(self, polygon_coords: List[List[float]]) -> List[float]:
        """
        Calculate bounding box from polygon coordinates
        
        Args:
            polygon_coords: List of [lat, lon] coordinate pairs
        
        Returns:
            Bounding box [min_lon, min_lat, max_lon, max_lat]
        """
        lats = [coord[0] for coord in polygon_coords]
        lons = [coord[1] for coord in polygon_coords]
        
        return [min(lons), min(lats), max(lons), max(lats)]
    
    def calculate_ndvi(self, stack) -> np.ndarray:
        """
        Calculate NDVI from Sentinel-2 bands
        
        NDVI = (NIR - Red) / (NIR + Red)
        
        Sentinel-2 bands:
        - Band 4 (B04): Red (665 nm)
        - Band 8 (B08): NIR (842 nm)
        
        Args:
            stack: Stacked Sentinel-2 data
        
        Returns:
            NDVI array
        """
        # Select bands (adjust indices based on your stack configuration)
        red = stack.sel(band='B04').values
        nir = stack.sel(band='B08').values
        
        # Calculate NDVI
        ndvi = (nir - red) / (nir + red + 1e-8)  # Add epsilon to avoid division by zero
        
        # Clip to valid range
        ndvi = np.clip(ndvi, -1, 1)
        
        return ndvi
    
    def calculate_ndwi(self, stack) -> np.ndarray:
        """
        Calculate NDWI from Sentinel-2 bands
        
        NDWI = (Green - NIR) / (Green + NIR)
        
        Sentinel-2 bands:
        - Band 3 (B03): Green (560 nm)
        - Band 8 (B08): NIR (842 nm)
        
        Args:
            stack: Stacked Sentinel-2 data
        
        Returns:
            NDWI array
        """
        green = stack.sel(band='B03').values
        nir = stack.sel(band='B08').values
        
        # Calculate NDWI
        ndwi = (green - nir) / (green + nir + 1e-8)
        
        # Clip to valid range
        ndwi = np.clip(ndwi, -1, 1)
        
        return ndwi
    
    def process_imagery(
        self,
        polygon_coords: List[List[float]],
        max_cloud_cover: int = 20
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Complete pipeline: Search, download, and process imagery
        
        Args:
            polygon_coords: Farm polygon coordinates [[lat, lon], ...]
            max_cloud_cover: Maximum acceptable cloud cover
        
        Returns:
            Tuple of (ndvi_array, ndwi_array)
        """
        # Get bounding box
        bbox = self.get_bbox_from_polygon(polygon_coords)
        print(f"📍 Bounding box: {bbox}")
        
        # Search for imagery
        items = self.search_imagery(bbox, max_cloud_cover=max_cloud_cover)
        
        if not items:
            raise ValueError("No imagery found for the specified area and date range")
        
        # Select the most recent cloud-free image
        items = sorted(items, key=lambda x: x.properties['eo:cloud_cover'])
        selected_item = items[0]
        
        print(f"📅 Using image from: {selected_item.datetime}")
        print(f"☁️ Cloud cover: {selected_item.properties['eo:cloud_cover']:.1f}%")
        
        # Sign the items (required for Microsoft Planetary Computer)
        signed_item = pc.sign(selected_item)
        
        # Stack the data
        stack = stackstac.stack(
            [signed_item],
            bounds_latlon=bbox,
            epsg=4326,
            resolution=10  # 10m resolution
        )
        
        # Calculate indices
        print("🔄 Calculating NDVI...")
        ndvi = self.calculate_ndvi(stack)
        
        print("🔄 Calculating NDWI...")
        ndwi = self.calculate_ndwi(stack)
        
        # Get the first timestep (most recent)
        ndvi = ndvi[0] if ndvi.ndim == 3 else ndvi
        ndwi = ndwi[0] if ndwi.ndim == 3 else ndwi
        
        print("✅ Processing complete!")
        
        return ndvi, ndwi


# Example usage function
def fetch_real_satellite_data(polygon_coords: List[List[float]]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convenience function to fetch real satellite data
    
    Args:
        polygon_coords: Farm polygon coordinates
    
    Returns:
        Tuple of (ndvi, ndwi) arrays
    
    Example:
        polygon = [[30.390, 30.335], [30.390, 30.348], [30.383, 30.348], [30.383, 30.335]]
        ndvi, ndwi = fetch_real_satellite_data(polygon)
    """
    try:
        fetcher = SentinelDataFetcher()
        ndvi, ndwi = fetcher.process_imagery(polygon_coords)
        return ndvi, ndwi
    except Exception as e:
        print(f"❌ Error fetching satellite data: {str(e)}")
        print("🔄 Falling back to demo mode...")
        
        # Return None to trigger demo mode in main app
        return None, None


# Integration instructions for app.py
"""
TO INTEGRATE INTO app.py:

1. Import this module:
   from sentinel_integration import fetch_real_satellite_data, SATELLITE_AVAILABLE

2. Replace demo data generation with real data:
   
   # In the analyze button callback:
   if not demo_mode and SATELLITE_AVAILABLE:
       # Try to fetch real data
       ndvi_data, ndwi_data = fetch_real_satellite_data(polygon_coords)
       
       if ndvi_data is None:
           # Fallback to demo
           st.warning("⚠️ Could not fetch satellite data. Using demo mode.")
           ndvi_data = processor.generate_demo_ndvi()
           ndwi_data = processor.generate_demo_ndwi()
   else:
       # Use demo data
       ndvi_data = processor.generate_demo_ndvi()
       ndwi_data = processor.generate_demo_ndwi()

3. Add requirements to requirements.txt:
   planetary-computer==1.0.0
   pystac-client==0.7.5
   stackstac==0.5.0
   rasterio==1.3.9

4. Test with a real farm location!
"""

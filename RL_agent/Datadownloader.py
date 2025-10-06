# Efficient GPS-Based Dataset Downloader (Based on Working Script)
import os
import requests
import numpy as np
import matplotlib.pyplot as plt
from sentinelhub import CRS, BBox, bbox_to_dimensions
import cv2
import json
from datetime import datetime
from typing import List, Dict, Optional
import time
from pathlib import Path

class EfficientGPSDownloader:
    """
    Downloads efficient dataset based on GPS location with multiple resolutions
    + Real drone imagery from Google Maps
    """
    
    def __init__(self, google_maps_api_key: Optional[str] = None, output_dir: str = "efficient_dataset"):
        # Your working credentials
        self.client_id = 'sh-29927974-7d1f-4f2e-9751-f59a8149a944'
        self.client_secret = 'Zqrx0QU4ivjOl3TM77zsm0Hhib8QWivO'
        self.base_url = 'https://sh.dataspace.copernicus.eu'
        self.token_url = 'https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token'
        
        # Google Maps API
        self.google_api_key = google_maps_api_key
        
        # Output directory
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Efficient resolution configs (reasonable file sizes)
        self.resolution_configs = [
            {"name": "low", "resolution_m": 10, "area_km": 5.0},      # ~5MB, good overview
            {"name": "medium", "resolution_m": 5, "area_km": 2.5},    # ~5MB, medium detail
            {"name": "high", "resolution_m": 3, "area_km": 1.5},      # ~5MB, high detail
        ]
        
        self.access_token = None
        self.dataset_info = {"downloads": [], "created_at": datetime.now().isoformat()}
        
        print(f"📡 Efficient GPS-Based Downloader")
        print(f"   Output: {self.output_dir}")
        print(f"   Resolutions: {[c['name'] for c in self.resolution_configs]}")
        print(f"   Google Maps: {'✅' if google_maps_api_key else '❌'}")
        
        self.get_access_token()
    
    def get_access_token(self):
        """Get access token using your working method"""
        print("🔑 Getting access token...")
        
        token_response = requests.post(
            self.token_url,
            data={
                'grant_type': 'client_credentials',
                'client_id': self.client_id,
                'client_secret': self.client_secret
            }
        )
        
        if token_response.status_code == 200:
            self.access_token = token_response.json()['access_token']
            print("✅ Access token obtained")
        else:
            raise Exception(f"Failed to get token: {token_response.status_code}")
    
    def download_multi_resolution_dataset(self, center_lat: float, center_lon: float, 
                                        location_name: str = "location") -> Dict:
        """
        Download same GPS location at multiple resolutions
        
        Args:
            center_lat, center_lon: GPS coordinates
            location_name: Name for this location (for file naming)
        """
        
        print(f"\n🎯 DOWNLOADING MULTI-RESOLUTION DATASET")
        print(f"   Location: {center_lat:.6f}°N, {center_lon:.6f}°E")
        print(f"   Name: {location_name}")
        
        downloaded_files = []
        
        for config in self.resolution_configs:
            print(f"\n📡 Downloading {config['name']} resolution...")
            print(f"   Resolution: {config['resolution_m']}m per pixel")
            print(f"   Area: {config['area_km']}km × {config['area_km']}km")
            
            try:
                file_info = self.download_satellite_image(
                    center_lat, center_lon, config, location_name
                )
                
                if file_info:
                    downloaded_files.append(file_info)
                    print(f"   ✅ Success: {file_info['filename']} ({file_info['file_size_mb']:.1f}MB)")
                else:
                    print(f"   ❌ Failed")
                
                time.sleep(2)  # Rate limiting
                
            except Exception as e:
                print(f"   ❌ Error: {e}")
        
        # Download Google Maps drone imagery for this area
        if self.google_api_key:
            print(f"\n📸 Downloading Google Maps drone imagery...")
            drone_images = self.download_google_drone_images(
                center_lat, center_lon, location_name
            )
            downloaded_files.extend(drone_images)
        
        # Save metadata
        location_metadata = {
            'location_name': location_name,
            'center_gps': [center_lat, center_lon],
            'download_time': datetime.now().isoformat(),
            'files': downloaded_files
        }
        
        self.dataset_info['downloads'].append(location_metadata)
        self.save_metadata()
        
        print(f"\n✅ Location '{location_name}' complete: {len(downloaded_files)} files")
        return location_metadata
    
    def download_satellite_image(self, center_lat: float, center_lon: float, 
                                config: Dict, location_name: str) -> Optional[Dict]:
        """Download single satellite image using your working method"""
        
        area_km = config['area_km']
        resolution_m = config['resolution_m']
        
        # Calculate bounds (your working method)
        offset_deg = area_km / 111.0 / 2
        bbox_coords = [
            center_lon - offset_deg * 1.3,  # lon_min (adjusted for latitude)
            center_lat - offset_deg,        # lat_min  
            center_lon + offset_deg * 1.3,  # lon_max
            center_lat + offset_deg         # lat_max
        ]
        
        bbox = BBox(bbox=bbox_coords, crs=CRS.WGS84)
        size = bbox_to_dimensions(bbox, resolution=resolution_m)
        
        # Your working evalscript (optimized for smaller files)
        evalscript = """
//VERSION=3
function setup() {
    return {
        input: ["B02", "B03", "B04", "B08"],
        output: { 
            bands: 4,
            sampleType: "UINT8"  // Use UINT8 for smaller files
        }
    };
}

function evaluatePixel(sample) {
    let gain = 3.0;
    let red = sample.B04 * gain;
    let green = sample.B03 * gain;
    let blue = sample.B02 * gain;
    let nir = sample.B08 * gain;
    
    return [
        Math.min(255, Math.max(0, red * 255)),
        Math.min(255, Math.max(0, green * 255)),
        Math.min(255, Math.max(0, blue * 255)),
        Math.min(255, Math.max(0, nir * 255))
    ];
}
"""
        
        # Your working request payload
        request_payload = {
            "input": {
                "bounds": {
                    "properties": {"crs": "http://www.opengis.net/def/crs/EPSG/0/4326"},
                    "bbox": bbox_coords
                },
                "data": [
                    {
                        "type": "sentinel-2-l2a",
                        "dataFilter": {
                            "timeRange": {
                                "from": "2024-06-01T00:00:00Z",
                                "to": "2024-09-30T23:59:59Z"
                            },
                            "mosaickingOrder": "leastCC"
                        }
                    }
                ]
            },
            "output": {
                "width": size[0],
                "height": size[1],
                "responses": [
                    {
                        "identifier": "default",
                        "format": {"type": "image/png"}  # PNG for better compression
                    }
                ]
            },
            "evalscript": evalscript
        }
        
        # Your working download method
        api_url = f"{self.base_url}/api/v1/process"
        headers = {
            'Authorization': f'Bearer {self.access_token}',
            'Content-Type': 'application/json'
        }
        
        response = requests.post(api_url, json=request_payload, headers=headers, timeout=120)
        
        if response.status_code == 200:
            # Create filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"sat_{location_name}_{config['name']}_{resolution_m}m_{timestamp}.png"
            filepath = self.output_dir / filename
            
            # Save file
            with open(filepath, 'wb') as f:
                f.write(response.content)
            
            file_size_mb = filepath.stat().st_size / 1024 / 1024
            
            return {
                'type': 'satellite',
                'filename': filename,
                'resolution_name': config['name'],
                'resolution_m': resolution_m,
                'area_km': area_km,
                'center_gps': [center_lat, center_lon],
                'bbox_coords': bbox_coords,
                'image_size': list(size),
                'file_size_mb': round(file_size_mb, 2),
                'download_time': timestamp
            }
        else:
            print(f"      API Error: {response.status_code}")
            return None
    
    def download_google_drone_images(self, center_lat: float, center_lon: float, 
                                   location_name: str, radius_km: float = 2.5) -> List[Dict]:
        """
        Download Google Maps images that simulate drone imagery
        """
        
        if not self.google_api_key:
            print("   ⚠️ No Google Maps API key provided")
            return []
        
        print(f"   📸 Getting Google Maps imagery...")
        
        drone_images = []
        
        # Different zoom levels and slight position offsets for variety
        image_configs = [
            {"zoom": 18, "size": "512x512", "offset": [0.0, 0.0], "name": "center_high"},
            {"zoom": 17, "size": "512x512", "offset": [0.001, 0.001], "name": "offset1_medium"},
            {"zoom": 19, "size": "512x512", "offset": [-0.001, 0.001], "name": "offset2_very_high"},
            {"zoom": 18, "size": "512x512", "offset": [0.001, -0.001], "name": "offset3_high"},
            {"zoom": 16, "size": "640x640", "offset": [0.0, 0.0], "name": "overview"},
        ]
        
        for i, img_config in enumerate(image_configs):
            try:
                # Calculate offset position
                offset_lat = center_lat + img_config['offset'][0]
                offset_lon = center_lon + img_config['offset'][1]
                
                # Google Maps Static API
                url = "https://maps.googleapis.com/maps/api/staticmap"
                params = {
                    'center': f"{offset_lat},{offset_lon}",
                    'zoom': img_config['zoom'],
                    'size': img_config['size'],
                    'maptype': 'satellite',
                    'key': self.google_api_key
                }
                
                response = requests.get(url, params=params, timeout=30)
                
                if response.status_code == 200:
                    # Save image
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"drone_{location_name}_{img_config['name']}_z{img_config['zoom']}_{timestamp}.jpg"
                    filepath = self.output_dir / filename
                    
                    with open(filepath, 'wb') as f:
                        f.write(response.content)
                    
                    file_size_mb = filepath.stat().st_size / 1024 / 1024
                    
                    # Apply drone-like transformations
                    self.apply_drone_transformations(filepath)
                    
                    drone_info = {
                        'type': 'drone_simulation',
                        'filename': filename,
                        'source': 'google_maps',
                        'zoom_level': img_config['zoom'],
                        'center_gps': [offset_lat, offset_lon],
                        'original_center': [center_lat, center_lon],
                        'size': img_config['size'],
                        'file_size_mb': round(file_size_mb, 2),
                        'config_name': img_config['name']
                    }
                    
                    drone_images.append(drone_info)
                    print(f"      ✅ {img_config['name']} (zoom {img_config['zoom']})")
                    
                else:
                    print(f"      ❌ Google Maps error: {response.status_code}")
                
                time.sleep(1)  # Rate limiting for Google Maps
                
            except Exception as e:
                print(f"      ❌ Error downloading {img_config['name']}: {e}")
        
        return drone_images
    
    def apply_drone_transformations(self, image_path: Path):
        """Apply realistic drone-like transformations to Google Maps images"""
        
        try:
            # Load image
            img = cv2.imread(str(image_path))
            if img is None:
                return
            
            # Random rotation (simulate drone orientation)
            h, w = img.shape[:2]
            angle = np.random.uniform(-15, 15)
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            img = cv2.warpAffine(img, M, (w, h))
            
            # Slight brightness/contrast variation
            alpha = np.random.uniform(0.9, 1.1)  # Contrast
            beta = np.random.randint(-10, 10)    # Brightness
            img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
            
            # Add subtle noise (atmospheric effects)
            noise = np.random.randint(-5, 5, img.shape, dtype=np.int16)
            img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            
            # Save transformed image
            cv2.imwrite(str(image_path), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            
        except Exception as e:
            print(f"      ⚠️ Transform error: {e}")
    
    def download_5km_area_dataset(self, center_lat: float, center_lon: float, 
                                 grid_size: int = 3) -> Dict:
        """
        Download 5km × 5km area as multiple overlapping locations
        
        Args:
            center_lat, center_lon: Center of 5km area
            grid_size: NxN grid of download points (default 3x3)
        """
        
        print(f"\n🗺️ DOWNLOADING 5km × 5km AREA DATASET")
        print(f"   Center: {center_lat:.6f}°N, {center_lon:.6f}°E")
        print(f"   Grid: {grid_size}×{grid_size} locations")
        print(f"   Total area: ~5km × 5km")
        
        # Calculate grid positions
        area_offset = 2.5 / 111.0  # 2.5km in degrees (half of 5km)
        grid_spacing = (2 * area_offset) / (grid_size - 1) if grid_size > 1 else 0
        
        all_downloads = []
        
        for i in range(grid_size):
            for j in range(grid_size):
                # Calculate grid position
                if grid_size == 1:
                    grid_lat = center_lat
                    grid_lon = center_lon
                else:
                    grid_lat = center_lat - area_offset + i * grid_spacing
                    grid_lon = center_lon - area_offset + j * grid_spacing
                
                location_name = f"grid_{i}_{j}"
                
                print(f"\n📍 Grid position ({i},{j}): {grid_lat:.6f}, {grid_lon:.6f}")
                
                # Download this location
                location_data = self.download_multi_resolution_dataset(
                    grid_lat, grid_lon, location_name
                )
                
                all_downloads.append(location_data)
        
        # Create area summary
        area_summary = {
            'area_center': [center_lat, center_lon],
            'area_size_km': 5.0,
            'grid_size': grid_size,
            'total_locations': len(all_downloads),
            'total_files': sum(len(loc['files']) for loc in all_downloads),
            'downloads': all_downloads,
            'created_at': datetime.now().isoformat()
        }
        
        # Save area summary
        summary_file = self.output_dir / f"area_summary_{center_lat:.4f}_{center_lon:.4f}.json"
        with open(summary_file, 'w') as f:
            json.dump(area_summary, f, indent=2)
        
        print(f"\n🎉 5km AREA DOWNLOAD COMPLETE!")
        print(f"   Locations: {len(all_downloads)}")
        print(f"   Total files: {area_summary['total_files']}")
        print(f"   Summary: {summary_file}")
        
        return area_summary
    
    def save_metadata(self):
        """Save dataset metadata"""
        metadata_file = self.output_dir / "dataset_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(self.dataset_info, f, indent=2)
    
    def visualize_dataset(self):
        """Visualize downloaded dataset"""
        
        print(f"\n🖼️ Visualizing dataset...")
        
        # Get all satellite images
        sat_images = []
        for download in self.dataset_info['downloads']:
            for file_info in download['files']:
                if file_info['type'] == 'satellite':
                    sat_images.append(file_info)
        
        if not sat_images:
            print("❌ No satellite images to visualize")
            return
        
        # Create visualization
        cols = min(3, len(sat_images))
        rows = (len(sat_images) + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(15, 5*rows))
        if rows == 1 and cols == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes.reshape(1, -1)
        
        for i, img_info in enumerate(sat_images):
            row = i // cols
            col = i % cols
            
            # Load and display image
            img_path = self.output_dir / img_info['filename']
            
            if img_path.exists():
                img = cv2.imread(str(img_path))
                if img is not None:
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    
                    if rows == 1:
                        ax = axes[col] if cols > 1 else axes[0]
                    else:
                        ax = axes[row, col]
                    
                    ax.imshow(img_rgb)
                    
                    title = f"{img_info['resolution_name']} ({img_info['resolution_m']}m)\n"
                    title += f"{img_info['area_km']}km area, {img_info['file_size_mb']}MB"
                    
                    ax.set_title(title, fontsize=10)
                    ax.axis('off')
        
        # Hide empty subplots
        for i in range(len(sat_images), rows * cols):
            row = i // cols
            col = i % cols
            if rows == 1:
                ax = axes[col] if cols > 1 else axes[0]
            else:
                ax = axes[row, col]
            ax.axis('off')
        
        plt.tight_layout()
        plt.show()
        
        print(f"✅ Visualization complete")

# Usage functions
def download_ukraine_efficient_dataset(google_api_key: Optional[str] = None):
    """Download efficient dataset for your Ukraine location"""
    
    print("🇺🇦" + "="*60 + "🇺🇦")
    print("    EFFICIENT UKRAINE DATASET DOWNLOAD")
    print("🇺🇦" + "="*60 + "🇺🇦")
    
    # Your coordinates
    ukraine_lat, ukraine_lon = 50.2957, 36.6619
    
    # Initialize downloader
    downloader = EfficientGPSDownloader(google_api_key, "ukraine_efficient_dataset")
    
    # Download 5km area with 3x3 grid
    area_data = downloader.download_5km_area_dataset(ukraine_lat, ukraine_lon, grid_size=3)
    
    # Visualize results
    downloader.visualize_dataset()
    
    print(f"\n📊 DATASET SUMMARY:")
    print(f"   Location: Ukraine ({ukraine_lat}, {ukraine_lon})")
    print(f"   Area coverage: 5km × 5km")
    print(f"   Grid points: 3×3 = 9 locations")
    print(f"   Resolution levels: 3 per location")
    print(f"   Drone simulations: 5 per location (if Google API available)")
    print(f"   Total files: ~{9 * 3 + (9 * 5 if google_api_key else 0)}")
    print(f"   Expected size: ~{9 * 3 * 5}MB satellite + {9 * 5 * 2 if google_api_key else 0}MB drone")
    
    return downloader, area_data

def quick_single_location_download(lat: float, lon: float, location_name: str = "test_location"):
    """Quick download for single location testing"""
    
    downloader = EfficientGPSDownloader(output_dir=f"quick_{location_name}")
    location_data = downloader.download_multi_resolution_dataset(lat, lon, location_name)
    downloader.visualize_dataset()
    
    return downloader, location_data

if __name__ == "__main__":
    print("📡 EFFICIENT GPS-BASED DATASET DOWNLOADER")
    print("="*60)
    print()
    print("🎯 FEATURES:")
    print("   • Based on your working Sentinel Hub script")
    print("   • Multiple resolutions for same GPS location")
    print("   • Reasonable file sizes (~5MB per image)")
    print("   • Google Maps drone simulation")
    print("   • 5km × 5km area coverage with grid sampling")
    print()
    print("🚀 USAGE:")
    print("   # Download Ukraine dataset")
    print("   downloader, data = download_ukraine_efficient_dataset(google_api_key)")
    print()
    print("   # Quick single location test")
    print("   downloader, data = quick_single_location_download(50.2957, 36.6619)")
    print()
    print("📁 OUTPUT STRUCTURE:")
    print("   efficient_dataset/")
    print("   ├── sat_grid_0_0_low_10m_*.png     (5MB satellite images)")
    print("   ├── sat_grid_0_0_medium_5m_*.png")
    print("   ├── sat_grid_0_0_high_3m_*.png")
    print("   ├── drone_grid_0_0_center_high_*.jpg (Google Maps)")
    print("   ├── ... (9 locations × 8 files each)")
    print("   └── dataset_metadata.json")
    print()
    print("🎉 Perfect for RL training with reasonable file sizes!")
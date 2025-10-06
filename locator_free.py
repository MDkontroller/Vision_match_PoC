import folium
import requests
import math
import numpy as np
from PIL import Image
import cv2
from io import BytesIO
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider
import tkinter as tk
from tkinter import messagebox, simpledialog

class SimpleOSMViewer:
    """
    Simple OpenStreetMap viewer with dynamic zoom for drone matching
    """
    
    def __init__(self, center_lat: float, center_lng: float):
        self.center_lat = center_lat
        self.center_lng = center_lng
        self.current_zoom = 15
        self.tile_cache = {}
        
    def deg2tile(self, lat: float, lng: float, zoom: int):
        """Convert lat/lng to tile coordinates"""
        lat_rad = math.radians(lat)
        n = 2.0 ** zoom
        x = (lng + 180.0) / 360.0 * n
        y = (1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n
        return int(x), int(y)
    
    def tile2deg(self, x: int, y: int, zoom: int):
        """Convert tile coordinates to lat/lng"""
        n = 2.0 ** zoom
        lng = x / n * 360.0 - 180.0
        lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * y / n)))
        lat = math.degrees(lat_rad)
        return lat, lng
    
    def download_osm_tile(self, x: int, y: int, zoom: int):
        """Download a single OSM tile"""
        tile_key = (x, y, zoom)
        
        if tile_key in self.tile_cache:
            return self.tile_cache[tile_key]
        
        # OSM tile URL
        url = f"https://tile.openstreetmap.org/{zoom}/{x}/{y}.png"
        
        headers = {
            'User-Agent': 'DroneLocator/1.0 (Educational Demo)'
        }
        
        try:
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code == 200:
                tile_image = Image.open(BytesIO(response.content))
                tile_array = np.array(tile_image)
                
                # Cache the tile
                self.tile_cache[tile_key] = tile_array
                return tile_array
            else:
                print(f"Failed to download tile {x},{y},{zoom}: {response.status_code}")
                return self.create_error_tile()
                
        except Exception as e:
            print(f"Error downloading tile: {e}")
            return self.create_error_tile()
    
    def create_error_tile(self):
        """Create a placeholder tile when download fails"""
        tile = np.full((256, 256, 3), 200, dtype=np.uint8)
        cv2.putText(tile, "No Data", (80, 128), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 100, 100), 2)
        return tile
    
    def get_area_image(self, center_lat: float, center_lng: float, 
                      zoom: int, grid_size: tuple = (4, 4)):
        """
        Download a grid of tiles to create larger area image
        grid_size: (width_tiles, height_tiles)
        """
        grid_w, grid_h = grid_size
        center_x, center_y = self.deg2tile(center_lat, center_lng, zoom)
        
        # Calculate tile range
        start_x = center_x - grid_w // 2
        start_y = center_y - grid_h // 2
        
        print(f"Downloading {grid_w}x{grid_h} tiles at zoom {zoom}...")
        
        # Download tiles
        tile_rows = []
        for y in range(grid_h):
            tile_row = []
            for x in range(grid_w):
                tile_x = start_x + x
                tile_y = start_y + y
                
                tile = self.download_osm_tile(tile_x, tile_y, zoom)
                tile_row.append(tile)
                
                print(f"  Downloaded tile ({x+1},{y+1})")
            
            # Combine tiles in this row
            row_image = np.hstack(tile_row)
            tile_rows.append(row_image)
        
        # Combine all rows
        full_image = np.vstack(tile_rows)
        
        # Calculate the actual geographic bounds
        top_left_lat, top_left_lng = self.tile2deg(start_x, start_y, zoom)
        bottom_right_lat, bottom_right_lng = self.tile2deg(start_x + grid_w, start_y + grid_h, zoom)
        
        bounds = {
            'north': top_left_lat,
            'south': bottom_right_lat,
            'west': top_left_lng,
            'east': bottom_right_lng,
            'center': (center_lat, center_lng),
            'zoom': zoom
        }
        
        print(f"✅ Created {full_image.shape[1]}x{full_image.shape[0]} map image")
        return full_image, bounds
    
    def create_interactive_map(self, save_file: str = "interactive_map.html"):
        """Create an interactive Folium map for area selection"""
        m = folium.Map(
            location=[self.center_lat, self.center_lng],
            zoom_start=self.current_zoom,
            tiles=None  # We'll add custom tiles
        )
        
        # Add different map layers
        folium.TileLayer(
            tiles='OpenStreetMap',
            name='OpenStreetMap',
            overlay=False,
            control=True
        ).add_to(m)
        
        folium.TileLayer(
            tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
            attr='Esri',
            name='Satellite',
            overlay=False,
            control=True
        ).add_to(m)
        
        # Add a marker at center
        folium.Marker(
            [self.center_lat, self.center_lng],
            popup=f"Search Center<br>Lat: {self.center_lat:.6f}<br>Lng: {self.center_lng:.6f}",
            tooltip="Search Center",
            icon=folium.Icon(color='red', icon='crosshairs')
        ).add_to(m)
        
        # Add search area rectangle (10km x 10km)
        # Rough conversion: 1 degree ≈ 111 km
        km_to_deg = 1 / 111
        
        bounds = [
            [self.center_lat - 5 * km_to_deg, self.center_lng - 5 * km_to_deg],  # SW
            [self.center_lat + 5 * km_to_deg, self.center_lng + 5 * km_to_deg]   # NE
        ]
        
        folium.Rectangle(
            bounds=bounds,
            color='blue',
            fill=True,
            fillOpacity=0.2,
            popup="10km x 10km Search Area"
        ).add_to(m)
        
        # Add click handler info
        folium.Marker(
            [self.center_lat + 3 * km_to_deg, self.center_lng + 3 * km_to_deg],
            popup="""
            <b>How to use:</b><br>
            1. Navigate to your area of interest<br>
            2. Note the coordinates<br>
            3. Use get_area_image() to download<br>
            4. Load your drone image<br>
            5. Find matches!
            """,
            icon=folium.Icon(color='green', icon='info-sign')
        ).add_to(m)
        
        # Add layer control
        folium.LayerControl().add_to(m)
        
        # Save map
        m.save(save_file)
        print(f"🗺️  Interactive map saved: {save_file}")
        print(f"📍 Open the map to select your search area")
        
        return m
    
    def create_zoomable_viewer(self, initial_image, bounds):
        """
        Create a matplotlib-based zoomable viewer
        """
        fig, ax = plt.subplots(figsize=(12, 10))
        fig.suptitle('🗺️ OSM Map Viewer - Click to Zoom', fontsize=14, fontweight='bold')
        
        # Display the image
        im = ax.imshow(initial_image)
        ax.set_title(f"Zoom Level: {bounds['zoom']} | Area: {bounds['north']:.4f} to {bounds['south']:.4f}")
        
        # Store current state
        self.current_image = initial_image
        self.current_bounds = bounds
        self.ax = ax
        self.im = im
        
        # Click handler for zooming
        def on_click(event):
            if event.inaxes == ax and event.dblclick:
                # Convert click coordinates to lat/lng
                h, w = self.current_image.shape[:2]
                click_x_ratio = event.xdata / w
                click_y_ratio = event.ydata / h
                
                # Calculate new center
                lat_range = self.current_bounds['north'] - self.current_bounds['south']
                lng_range = self.current_bounds['east'] - self.current_bounds['west']
                
                new_lat = self.current_bounds['south'] + (1 - click_y_ratio) * lat_range
                new_lng = self.current_bounds['west'] + click_x_ratio * lng_range
                
                # Zoom in
                new_zoom = min(18, self.current_bounds['zoom'] + 2)
                
                print(f"🔍 Zooming to: {new_lat:.6f}, {new_lng:.6f} at zoom {new_zoom}")
                
                # Download new image
                try:
                    new_image, new_bounds = self.get_area_image(new_lat, new_lng, new_zoom, (3, 3))
                    
                    # Update display
                    self.im.set_array(new_image)
                    self.im.set_extent([0, new_image.shape[1], new_image.shape[0], 0])
                    ax.set_title(f"Zoom Level: {new_bounds['zoom']} | Center: {new_lat:.6f}, {new_lng:.6f}")
                    
                    # Store new state
                    self.current_image = new_image
                    self.current_bounds = new_bounds
                    
                    fig.canvas.draw()
                    
                except Exception as e:
                    print(f"❌ Zoom failed: {e}")
        
        fig.canvas.mpl_connect('button_press_event', on_click)
        
        # Add instructions
        ax.text(0.02, 0.98, "Double-click to zoom in\nCurrent area coverage shown", 
                transform=ax.transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.show()
        
        return fig
    
    def save_current_area(self, filename: str = "search_area.png"):
        """Save the current area as an image for drone matching"""
        if hasattr(self, 'current_image'):
            # Convert RGB to BGR for OpenCV
            bgr_image = cv2.cvtColor(self.current_image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(filename, bgr_image)
            print(f"💾 Search area saved: {filename}")
            print(f"📏 Image size: {self.current_image.shape}")
            print(f"📍 Geographic bounds:")
            print(f"   North: {self.current_bounds['north']:.6f}")
            print(f"   South: {self.current_bounds['south']:.6f}")
            print(f"   West: {self.current_bounds['west']:.6f}")
            print(f"   East: {self.current_bounds['east']:.6f}")
            
            return filename, self.current_bounds
        else:
            print("❌ No area loaded yet. Get an area first!")
            return None, None

def quick_demo():
    """Quick demonstration of the OSM viewer"""
    print("🗺️  OpenStreetMap Area Viewer Demo")
    print("="*40)
    
    # Example coordinates (you can change these)
    lat, lng = 40.7829, -73.9654  # Central Park, NYC
    
    print(f"📍 Center: {lat}, {lng}")
    
    # Create viewer
    viewer = SimpleOSMViewer(lat, lng)
    
    # Create interactive web map for area selection
    viewer.create_interactive_map("area_selection.html")
    
    # Download initial area (4x4 tiles at zoom 15)
    print("\n📡 Downloading initial area...")
    image, bounds = viewer.get_area_image(lat, lng, zoom=15, grid_size=(4, 4))
    
    # Create zoomable viewer
    print("\n🔍 Opening zoomable viewer...")
    print("💡 Double-click on the map to zoom into that area")
    viewer.create_zoomable_viewer(image, bounds)
    
    # Save the area for later use
    viewer.save_current_area("my_search_area.png")
    
    print("\n✅ Demo complete!")
    print("📁 Files created:")
    print("   - area_selection.html (interactive map)")
    print("   - my_search_area.png (downloaded map area)")

def simple_usage_example():
    """Show simple usage for drone matching"""
    print("\n🚁 Simple Usage for Drone Matching:")
    print("="*40)
    
    code_example = """
# 1. Set your search area
viewer = SimpleOSMViewer(lat=40.7829, lng=-73.9654)

# 2. Download map data (adjust zoom and grid_size as needed)
map_image, bounds = viewer.get_area_image(
    center_lat=40.7829, 
    center_lng=-73.9654,
    zoom=16,           # Higher zoom = more detail
    grid_size=(3, 3)   # 3x3 tiles = larger area
)

# 3. Save for drone matching
cv2.imwrite("satellite_reference.png", map_image)

# 4. Load your drone image and match
drone_img = cv2.imread("your_drone_photo.jpg")
# ... run your SIFT matching algorithm ...

# 5. Convert pixel coordinates back to GPS
def pixel_to_gps(pixel_x, pixel_y, bounds, image_shape):
    h, w = image_shape[:2]
    lat_range = bounds['north'] - bounds['south']
    lng_range = bounds['east'] - bounds['west']
    
    lat = bounds['south'] + (1 - pixel_y/h) * lat_range
    lng = bounds['west'] + (pixel_x/w) * lng_range
    return lat, lng

# Found drone at pixel (x, y)? Convert to GPS:
drone_lat, drone_lng = pixel_to_gps(found_x, found_y, bounds, map_image.shape)
print(f"Drone location: {drone_lat}, {drone_lng}")
    """
    
    print(code_example)

if __name__ == "__main__":
    print("🗺️  Simple OSM Map Viewer for Drone Matching")
    print("="*50)
    print("This tool helps you:")
    print("• Download map areas from OpenStreetMap")
    print("• Zoom dynamically to find the right detail level")
    print("• Save areas for drone photo matching")
    print("• No API keys required!")
    print()
    
    # Show usage
    simple_usage_example()
    
    print("\n💡 Ready to run demo?")
    print("Call: quick_demo()")
    
    # Uncomment to run immediately:
    quick_demo()
    
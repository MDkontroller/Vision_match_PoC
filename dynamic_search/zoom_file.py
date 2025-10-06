# Robust TIFF loader that handles Sentinel Hub TIFF files
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
import struct
import os

class RobustTIFFLoader:
    """
    Robust TIFF loader that can handle multi-band satellite TIFF files
    """
    
    def __init__(self, tiff_path):
        self.tiff_path = tiff_path
        self.load_tiff_data()
        self.setup_coordinates()
    
    def load_tiff_data(self):
        """Load TIFF data using multiple methods"""
        print(f"Loading TIFF: {self.tiff_path}")
        
        # Method 1: Try with PIL/Pillow
        try:
            from PIL import Image
            pil_image = Image.open(self.tiff_path)
            self.image = np.array(pil_image)
            print(f"✅ Loaded with PIL: {self.image.shape}, dtype: {self.image.dtype}")
            return
        except Exception as e:
            print(f"PIL failed: {e}")
        
        # Method 2: Try with imageio
        try:
            import imageio
            self.image = imageio.imread(self.tiff_path)
            print(f"✅ Loaded with imageio: {self.image.shape}, dtype: {self.image.dtype}")
            return
        except ImportError:
            print("imageio not available, trying next method...")
        except Exception as e:
            print(f"imageio failed: {e}")
        
        # Method 3: Try with skimage
        try:
            from skimage import io
            self.image = io.imread(self.tiff_path)
            print(f"✅ Loaded with skimage: {self.image.shape}, dtype: {self.image.dtype}")
            return
        except ImportError:
            print("skimage not available, trying next method...")
        except Exception as e:
            print(f"skimage failed: {e}")
        
        # Method 4: Try with tifffile (most robust for scientific TIFF)
        try:
            import tifffile
            self.image = tifffile.imread(self.tiff_path)
            print(f"✅ Loaded with tifffile: {self.image.shape}, dtype: {self.image.dtype}")
            return
        except ImportError:
            print("tifffile not available, trying raw read...")
        except Exception as e:
            print(f"tifffile failed: {e}")
        
        # Method 5: Raw binary read (last resort)
        try:
            self.load_raw_tiff()
            return
        except Exception as e:
            print(f"Raw read failed: {e}")
        
        raise Exception("Could not load TIFF file with any available method")
    
    def load_raw_tiff(self):
        """Load TIFF as raw binary data (basic approach)"""
        print("Attempting raw TIFF read...")
        
        with open(self.tiff_path, 'rb') as f:
            # Read TIFF header
            header = f.read(8)
            
            # Check if it's a valid TIFF
            if header[:2] in [b'II', b'MM']:  # Little or big endian
                print("✅ Valid TIFF header detected")
                
                # For simplicity, try to estimate dimensions from file size
                file_size = os.path.getsize(self.tiff_path)
                
                # Estimate square image size (this is a rough guess)
                estimated_pixels = file_size // 8  # Assume 16-bit per band, 4 bands
                estimated_size = int(np.sqrt(estimated_pixels))
                
                print(f"File size: {file_size} bytes")
                print(f"Estimated size: {estimated_size} x {estimated_size}")
                
                # Read raw data
                f.seek(0)
                raw_data = f.read()
                
                # Try to interpret as 16-bit data
                data_16bit = np.frombuffer(raw_data, dtype=np.uint16)
                
                # Try different reshapes
                for size in [estimated_size, estimated_size-1, estimated_size+1]:
                    try:
                        total_elements = size * size * 4  # 4 bands
                        if len(data_16bit) >= total_elements:
                            reshaped = data_16bit[:total_elements].reshape(size, size, 4)
                            self.image = reshaped
                            print(f"✅ Raw read successful: {self.image.shape}")
                            return
                    except:
                        continue
                
                # If reshape failed, try as 2D
                sqrt_len = int(np.sqrt(len(data_16bit)))
                self.image = data_16bit[:sqrt_len*sqrt_len].reshape(sqrt_len, sqrt_len)
                print(f"✅ Raw read as 2D: {self.image.shape}")
            else:
                raise Exception("Not a valid TIFF file")
    
    def setup_coordinates(self):
        """Setup coordinate system"""
        # Your known parameters
        center_lat, center_lon = 50.2957, 36.6619
        
        # Estimate area size from image dimensions
        if len(self.image.shape) >= 2:
            height, width = self.image.shape[:2]
            estimated_area_km = max(width, height) * 10 / 1000  # Assume 10m resolution
        else:
            estimated_area_km = 10  # Default
        
        # Calculate bounds
        offset_deg = estimated_area_km / 111.0 / 2
        
        self.bounds = [
            center_lon - offset_deg * 1.3,  # lon_min
            center_lat - offset_deg,        # lat_min  
            center_lon + offset_deg * 1.3,  # lon_max
            center_lat + offset_deg         # lat_max
        ]
        
        self.center = [center_lat, center_lon]
        self.area_km = estimated_area_km
        
        print(f"✅ Estimated area: {estimated_area_km:.1f}km")
        print(f"✅ Bounds: {self.bounds}")

class SatelliteZoomInterface:
    """
    Zoom interface for satellite images
    """
    
    def __init__(self, image_data, bounds, center, area_km):
        self.original_image = image_data
        self.bounds = bounds
        self.center = center
        self.area_km = area_km
        self.prepare_display_image()
        self.create_interface()
    
    def prepare_display_image(self):
        """Prepare image for display"""
        print(f"Preparing display image from: {self.original_image.shape}")
        
        # Handle different image formats
        if len(self.original_image.shape) == 3:
            # Multi-band image
            if self.original_image.shape[2] >= 3:
                # Use first 3 bands as RGB
                rgb_data = self.original_image[:, :, :3]
            else:
                # Duplicate single band to RGB
                rgb_data = np.stack([self.original_image[:, :, 0]] * 3, axis=-1)
        elif len(self.original_image.shape) == 2:
            # Single band - convert to RGB
            rgb_data = np.stack([self.original_image] * 3, axis=-1)
        else:
            raise ValueError(f"Unsupported image shape: {self.original_image.shape}")
        
        # Normalize for display
        if rgb_data.dtype == np.uint16:
            # 16-bit data - normalize to 8-bit
            self.display_image = (rgb_data.astype(np.float32) / 65535.0 * 255).astype(np.uint8)
        elif rgb_data.dtype in [np.float32, np.float64]:
            # Float data - assume 0-1 range
            self.display_image = (np.clip(rgb_data, 0, 1) * 255).astype(np.uint8)
        else:
            # Already 8-bit or other
            self.display_image = rgb_data.astype(np.uint8)
        
        print(f"✅ Display image ready: {self.display_image.shape}, range: {self.display_image.min()}-{self.display_image.max()}")
    
    def create_interface(self):
        """Create zoom interface"""
        
        # Create figure
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(20, 10))
        
        # Left panel - full image
        extent = [self.bounds[0], self.bounds[2], self.bounds[1], self.bounds[3]]
        self.im1 = self.ax1.imshow(self.display_image, extent=extent, aspect='auto')
        
        self.ax1.set_title(f"Sentinel-2 Ukraine - {self.area_km:.1f}km Area")
        self.ax1.set_xlabel('Longitude (°)')
        self.ax1.set_ylabel('Latitude (°)')
        self.ax1.grid(True, alpha=0.3)
        
        # Mark center
        center_lat, center_lon = self.center
        self.ax1.plot(center_lon, center_lat, 'r+', markersize=15, markeredgewidth=3,
                     label=f'Target: {center_lat:.4f}°N, {center_lon:.4f}°E')
        self.ax1.legend()
        
        # Right panel - zoom
        self.im2 = self.ax2.imshow(self.display_image)
        self.ax2.set_title("Select area on left to zoom")
        self.ax2.axis('off')
        
        # Rectangle selector
        self.selector = RectangleSelector(
            self.ax1, self.on_select,
            useblit=True,
            button=[1],
            minspanx=0.001, minspany=0.001,
            spancoords='data',
            interactive=True
        )
        
        # Info
        info = f"""
        IMAGE INFO:
        • Original shape: {self.original_image.shape}
        • Display shape: {self.display_image.shape}
        • Data type: {self.original_image.dtype}
        • Estimated area: {self.area_km:.1f}km × {self.area_km:.1f}km
        • Center: {center_lat:.4f}°N, {center_lon:.4f}°E
        
        USAGE:
        • Click and drag on LEFT image to zoom
        • Selected area shows on RIGHT
        • Use toolbar for pan/zoom
        """
        
        self.fig.text(0.02, 0.02, info, fontsize=9,
                     bbox=dict(boxstyle="round", facecolor='lightblue', alpha=0.8))
    
    def on_select(self, eclick, erelease):
        """Handle zoom selection"""
        
        # Get coordinates
        lon1, lat1 = eclick.xdata, eclick.ydata
        lon2, lat2 = erelease.xdata, erelease.ydata
        
        if None in [lon1, lat1, lon2, lat2]:
            return
        
        lon_min, lon_max = min(lon1, lon2), max(lon1, lon2)
        lat_min, lat_max = min(lat1, lat2), max(lat1, lat2)
        
        # Convert to pixels
        h, w = self.display_image.shape[:2]
        
        x1 = int((lon_min - self.bounds[0]) / (self.bounds[2] - self.bounds[0]) * w)
        x2 = int((lon_max - self.bounds[0]) / (self.bounds[2] - self.bounds[0]) * w)
        y1 = int((self.bounds[3] - lat_max) / (self.bounds[3] - self.bounds[1]) * h)
        y2 = int((self.bounds[3] - lat_min) / (self.bounds[3] - self.bounds[1]) * h)
        
        # Clamp to bounds
        x1, x2 = max(0, min(x1, x2)), min(w, max(x1, x2))
        y1, y2 = max(0, min(y1, y2)), min(h, max(y1, y2))
        
        if x2 - x1 < 5 or y2 - y1 < 5:
            return
        
        # Extract zoom
        zoomed = self.display_image[y1:y2, x1:x2]
        
        # Update display
        self.im2.set_array(zoomed)
        self.im2.set_extent([0, zoomed.shape[1], zoomed.shape[0], 0])
        
        # Calculate info
        area_w_km = (lon_max - lon_min) * 111
        area_h_km = (lat_max - lat_min) * 111
        
        title = f"ZOOM: {area_w_km:.2f} × {area_h_km:.2f} km\n"
        title += f"Coords: {lon_min:.5f}°E to {lon_max:.5f}°E\n"
        title += f"        {lat_min:.5f}°N to {lat_max:.5f}°N\n"
        title += f"Pixels: {zoomed.shape[1]} × {zoomed.shape[0]}"
        
        self.ax2.set_title(title, fontsize=10)
        self.ax2.axis('on')
        
        self.fig.canvas.draw()
        
        print(f"\n=== ZOOM INFO ===")
        print(f"Area: {area_w_km:.2f} × {area_h_km:.2f} km")
        print(f"Coords: [{lon_min:.6f}, {lat_min:.6f}, {lon_max:.6f}, {lat_max:.6f}]")
        print(f"Pixels: {zoomed.shape}")
    
    def show(self):
        """Show interface"""
        plt.tight_layout()
        plt.show()
        return self

# Main function to load and zoom
def load_and_zoom(tiff_path):
    """Load TIFF and create zoom interface"""
    
    try:
        # Load TIFF data
        loader = RobustTIFFLoader(tiff_path)
        
        # Create zoom interface
        zoomer = SatelliteZoomInterface(
            loader.image, 
            loader.bounds, 
            loader.center, 
            loader.area_km
        )
        
        print(f"\n🎯 ZOOM INTERFACE READY!")
        print("Click and drag on the LEFT image to select zoom areas!")
        
        zoomer.show()
        return zoomer
        
    except Exception as e:
        print(f"❌ Failed to load {tiff_path}: {e}")
        return None

# Try to load your TIFF file
print("=== ROBUST TIFF LOADER ===")
print("Attempting to load your satellite image...")

# First install imageio if needed (it's more robust than PIL for TIFF)
try:
    import imageio
    print("✅ imageio available")
except ImportError:
    print("Installing imageio for better TIFF support...")
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "imageio"])
    import imageio

# Load your file
zoomer = load_and_zoom('sentinel2_ukraine_10km.tif')

if zoomer is None:
    print("\n🔧 ALTERNATIVE: If the TIFF still won't load, try:")
    print("1. Check if you have a PNG file instead:")
    png_files = [f for f in os.listdir('.') if f.endswith('.png') and 'sentinel' in f.lower()]
    if png_files:
        print(f"   Found PNG files: {png_files}")
        print(f"   Try: zoomer = load_and_zoom('{png_files[0]}')")
    
    print("2. Or use the working PNG file:")
    if os.path.exists('sentinel2_ukraine_direct_api.png'):
        print("   zoomer = load_and_zoom('sentinel2_ukraine_direct_api.png')")
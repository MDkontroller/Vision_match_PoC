# Examine existing TIF file and extract drone-like crops
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import json
import os
from pathlib import Path
from typing import List, Dict, Tuple
import random

class TIFDroneExtractor:
    """
    Examine existing TIF file and extract drone-like training crops
    """
    
    def __init__(self, tif_path: str = "sentinel2_ukraine_10km.TIF"):
        self.tif_path = tif_path
        self.image = None
        self.image_info = {}
        self.extracted_crops = []
        
        # Your known location info
        self.center_lat = 50.2957
        self.center_lon = 36.6619
        self.area_km = 10
        
        print(f"📁 TIF Drone Extractor")
        print(f"   Input file: {tif_path}")
        
        self.load_and_analyze_tif()
    
    def load_and_analyze_tif(self):
        """Load and analyze the TIF file"""
        
        if not os.path.exists(self.tif_path):
            print(f"❌ File not found: {self.tif_path}")
            return
        
        print(f"\n📊 ANALYZING TIF FILE...")
        
        try:
            # Try multiple loading methods
            methods = [
                ("PIL", self.load_with_pil),
                ("OpenCV", self.load_with_opencv),
                ("Raw", self.load_with_raw_tiff)
            ]
            
            for method_name, method_func in methods:
                print(f"   Trying {method_name}...")
                try:
                    result = method_func()
                    if result is not None:
                        self.image = result
                        print(f"   ✅ Loaded with {method_name}")
                        break
                except Exception as e:
                    print(f"   ❌ {method_name} failed: {e}")
            
            if self.image is not None:
                self.analyze_image()
            else:
                print("❌ Could not load TIF file with any method")
                
        except Exception as e:
            print(f"❌ Error analyzing TIF: {e}")
    
    def load_with_pil(self):
        """Load TIF with PIL"""
        img = Image.open(self.tif_path)
        return np.array(img)
    
    def load_with_opencv(self):
        """Load TIF with OpenCV"""
        return cv2.imread(self.tif_path, cv2.IMREAD_UNCHANGED)
    
    def load_with_raw_tiff(self):
        """Load TIF with basic file reading"""
        try:
            import tifffile
            return tifffile.imread(self.tif_path)
        except ImportError:
            # Fallback to PIL if tifffile not available
            return self.load_with_pil()
    
    def analyze_image(self):
        """Analyze the loaded image"""
        
        print(f"\n📊 IMAGE ANALYSIS:")
        print(f"   Shape: {self.image.shape}")
        print(f"   Data type: {self.image.dtype}")
        print(f"   Value range: {self.image.min()} - {self.image.max()}")
        
        # Store image info
        self.image_info = {
            'shape': self.image.shape,
            'dtype': str(self.image.dtype),
            'value_range': [int(self.image.min()), int(self.image.max())],
            'file_size_mb': os.path.getsize(self.tif_path) / 1024 / 1024,
            'center_gps': [self.center_lat, self.center_lon],
            'area_km': self.area_km
        }
        
        print(f"   File size: {self.image_info['file_size_mb']:.1f} MB")
        
        # Calculate ground resolution
        if len(self.image.shape) >= 2:
            height, width = self.image.shape[:2]
            area_m = self.area_km * 1000
            resolution_m = area_m / width  # Approximate
            self.image_info['estimated_resolution_m'] = resolution_m
            print(f"   Estimated resolution: {resolution_m:.1f}m per pixel")
        
        # Determine if it's multi-band
        if len(self.image.shape) == 3:
            bands = self.image.shape[2]
            print(f"   Bands: {bands}")
            self.image_info['bands'] = bands
        else:
            print(f"   Bands: 1 (grayscale)")
            self.image_info['bands'] = 1
    
    def display_overview(self):
        """Display overview of the TIF file"""
        
        if self.image is None:
            print("❌ No image loaded")
            return
        
        print(f"\n🖼️ DISPLAYING IMAGE OVERVIEW...")
        
        # Prepare image for display
        display_image = self.prepare_for_display(self.image)
        
        # Create overview plot
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Ukraine Satellite Image Analysis\n{self.tif_path}', fontsize=16)
        
        # Main image
        axes[0, 0].imshow(display_image)
        axes[0, 0].set_title(f'Full Image ({self.image.shape[1]}×{self.image.shape[0]})')
        axes[0, 0].axis('off')
        
        # Mark center point
        center_x, center_y = self.image.shape[1] // 2, self.image.shape[0] // 2
        axes[0, 0].plot(center_x, center_y, 'r+', markersize=20, markeredgewidth=3)
        axes[0, 0].text(center_x + 20, center_y - 20, 
                       f'Center\n{self.center_lat:.4f}°N\n{self.center_lon:.4f}°E',
                       color='red', fontweight='bold', fontsize=10,
                       bbox=dict(boxstyle="round", facecolor='white', alpha=0.8))
        
        # Histogram
        if len(display_image.shape) == 3:
            colors = ['red', 'green', 'blue']
            for i, color in enumerate(colors):
                if i < display_image.shape[2]:
                    axes[0, 1].hist(display_image[:, :, i].flatten(), bins=50, alpha=0.5, 
                                   color=color, label=f'Band {i+1}')
        else:
            axes[0, 1].hist(display_image.flatten(), bins=50, alpha=0.7, color='gray')
        
        axes[0, 1].set_title('Pixel Value Distribution')
        axes[0, 1].set_xlabel('Pixel Value')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].legend()
        
        # Image info
        info_text = f"""
IMAGE INFORMATION:
• File: {os.path.basename(self.tif_path)}
• Size: {self.image_info['file_size_mb']:.1f} MB
• Dimensions: {self.image.shape[1]} × {self.image.shape[0]}
• Bands: {self.image_info['bands']}
• Data Type: {self.image_info['dtype']}
• Value Range: {self.image_info['value_range'][0]} - {self.image_info['value_range'][1]}
• Est. Resolution: {self.image_info.get('estimated_resolution_m', 'Unknown'):.1f}m/pixel
• Location: {self.center_lat:.4f}°N, {self.center_lon:.4f}°E
• Coverage: {self.area_km}km × {self.area_km}km
        """
        
        axes[1, 0].text(0.05, 0.95, info_text.strip(), transform=axes[1, 0].transAxes,
                        fontsize=10, verticalalignment='top', fontfamily='monospace',
                        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        axes[1, 0].set_title('Image Information')
        axes[1, 0].axis('off')
        
        # Sample crops preview
        sample_crops = self.extract_sample_crops(num_samples=4, crop_size=128)
        
        if sample_crops:
            # Show sample crops in a grid
            crop_grid = np.hstack([np.vstack(sample_crops[:2]), np.vstack(sample_crops[2:4])])
            axes[1, 1].imshow(crop_grid)
            axes[1, 1].set_title('Sample 128×128 Crops\n(Potential Drone Views)')
            axes[1, 1].axis('off')
        else:
            axes[1, 1].text(0.5, 0.5, 'Could not extract\nsample crops', 
                           ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('Sample Crops')
        
        plt.tight_layout()
        plt.show()
    
    def prepare_for_display(self, image):
        """Prepare image for matplotlib display"""
        
        if len(image.shape) == 2:
            # Grayscale
            return image
        
        # Multi-band image
        if image.shape[2] >= 3:
            # Use first 3 bands as RGB
            display_img = image[:, :, :3].copy()
        else:
            # Convert single band to RGB
            display_img = np.stack([image[:, :, 0]] * 3, axis=-1)
        
        # Normalize for display
        if display_img.dtype == np.uint16:
            display_img = (display_img.astype(np.float32) / 65535.0 * 255).astype(np.uint8)
        elif display_img.dtype in [np.float32, np.float64]:
            display_img = (np.clip(display_img, 0, 1) * 255).astype(np.uint8)
        
        return display_img
    
    def extract_sample_crops(self, num_samples: int = 4, crop_size: int = 128):
        """Extract sample crops for preview"""
        
        if self.image is None:
            return []
        
        display_img = self.prepare_for_display(self.image)
        h, w = display_img.shape[:2]
        
        crops = []
        margin = crop_size // 2
        
        for _ in range(num_samples):
            # Random crop position (avoid edges)
            x = random.randint(margin, w - margin - crop_size)
            y = random.randint(margin, h - margin - crop_size)
            
            crop = display_img[y:y+crop_size, x:x+crop_size]
            crops.append(crop)
        
        return crops
    
    def extract_drone_training_crops(self, num_crops: int = 50, crop_sizes: List[int] = [128, 256, 512]):
        """
        Extract realistic drone training crops from the satellite image
        """
        
        if self.image is None:
            print("❌ No image loaded")
            return []
        
        print(f"\n✂️ EXTRACTING DRONE TRAINING CROPS")
        print(f"   Number of crops: {num_crops}")
        print(f"   Crop sizes: {crop_sizes}")
        
        output_dir = Path("drone_crops_from_tif")
        output_dir.mkdir(exist_ok=True)
        
        display_img = self.prepare_for_display(self.image)
        h, w = display_img.shape[:2]
        
        all_crops = []
        
        for crop_idx in range(num_crops):
            # Random crop size
            crop_size = random.choice(crop_sizes)
            margin = crop_size // 2
            
            if w < crop_size + 2*margin or h < crop_size + 2*margin:
                print(f"   ⚠️ Image too small for crop size {crop_size}")
                continue
            
            # Random position (avoid edges)
            center_x = random.randint(margin, w - margin)
            center_y = random.randint(margin, h - margin)
            
            x1 = center_x - crop_size // 2
            y1 = center_y - crop_size // 2
            x2 = x1 + crop_size
            y2 = y1 + crop_size
            
            # Extract crop
            crop = display_img[y1:y2, x1:x2].copy()
            
            # Apply drone-like transformations
            transformed_crop = self.apply_drone_transformations(crop)
            
            # Calculate GPS coordinates for this crop
            crop_gps = self.pixel_to_gps(center_x, center_y)
            
            # Save crop
            filename = f"drone_crop_{crop_idx:03d}_{crop_size}px.jpg"
            filepath = output_dir / filename
            cv2.imwrite(str(filepath), transformed_crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
            
            # Store metadata
            crop_info = {
                'filename': filename,
                'crop_index': crop_idx,
                'crop_size': crop_size,
                'center_pixel': [center_x, center_y],
                'crop_bounds_pixel': [x1, y1, x2, y2],
                'estimated_gps': crop_gps,
                'source_image': self.tif_path,
                'transformation_applied': True
            }
            
            all_crops.append(crop_info)
            
            if crop_idx % 10 == 0:
                print(f"   Progress: {crop_idx}/{num_crops}")
        
        # Save metadata
        metadata_file = output_dir / "crops_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump({
                'source_image': self.tif_path,
                'source_info': self.image_info,
                'total_crops': len(all_crops),
                'crop_sizes': crop_sizes,
                'crops': all_crops
            }, f, indent=2)
        
        self.extracted_crops = all_crops
        
        print(f"✅ Extracted {len(all_crops)} drone crops")
        print(f"   Output directory: {output_dir}")
        print(f"   Metadata: {metadata_file}")
        
        return all_crops
    
    def apply_drone_transformations(self, crop: np.ndarray) -> np.ndarray:
        """Apply realistic drone-like transformations"""
        
        # Random rotation (-20 to +20 degrees)
        h, w = crop.shape[:2]
        angle = random.uniform(-20, 20)
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        crop = cv2.warpAffine(crop, M, (w, h))
        
        # Random brightness and contrast (atmospheric conditions)
        alpha = random.uniform(0.8, 1.2)  # Contrast
        beta = random.randint(-20, 20)    # Brightness
        crop = cv2.convertScaleAbs(crop, alpha=alpha, beta=beta)
        
        # Add atmospheric blur
        if random.random() < 0.3:  # 30% chance
            blur_size = random.choice([3, 5])
            crop = cv2.GaussianBlur(crop, (blur_size, blur_size), 0)
        
        # Add noise
        if random.random() < 0.4:  # 40% chance
            noise = np.random.randint(-15, 15, crop.shape, dtype=np.int16)
            crop = np.clip(crop.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        # Random scale (simulate altitude changes)
        if random.random() < 0.3:  # 30% chance
            scale = random.uniform(0.8, 1.2)
            h_new, w_new = int(h * scale), int(w * scale)
            crop_scaled = cv2.resize(crop, (w_new, h_new))
            
            # Crop or pad to original size
            if scale > 1.0:
                # Crop center
                start_x = (w_new - w) // 2
                start_y = (h_new - h) // 2
                crop = crop_scaled[start_y:start_y+h, start_x:start_x+w]
            else:
                # Pad with border
                crop = np.zeros((h, w, crop.shape[2]), dtype=crop.dtype)
                start_x = (w - w_new) // 2
                start_y = (h - h_new) // 2
                crop[start_y:start_y+h_new, start_x:start_x+w_new] = crop_scaled
        
        return crop
    
    def pixel_to_gps(self, pixel_x: int, pixel_y: int) -> List[float]:
        """Convert pixel coordinates to estimated GPS coordinates"""
        
        if self.image is None:
            return [0.0, 0.0]
        
        h, w = self.image.shape[:2]
        
        # Calculate offsets from center
        center_x, center_y = w // 2, h // 2
        offset_x = pixel_x - center_x
        offset_y = center_y - pixel_y  # Y is flipped
        
        # Convert to degrees (rough approximation)
        area_deg = self.area_km / 111.0
        pixel_to_deg_x = area_deg / w
        pixel_to_deg_y = area_deg / h
        
        # Calculate GPS coordinates
        estimated_lon = self.center_lon + offset_x * pixel_to_deg_x
        estimated_lat = self.center_lat + offset_y * pixel_to_deg_y
        
        return [estimated_lat, estimated_lon]
    
    def visualize_extracted_crops(self, num_display: int = 12):
        """Visualize extracted drone crops"""
        
        if not self.extracted_crops:
            print("❌ No crops extracted yet")
            return
        
        print(f"🖼️ Visualizing {num_display} extracted crops...")
        
        # Load sample crops
        output_dir = Path("drone_crops_from_tif")
        
        sample_crops = self.extracted_crops[:num_display]
        
        cols = 4
        rows = (len(sample_crops) + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(16, 4*rows))
        if rows == 1:
            axes = axes.reshape(1, -1)
        
        for i, crop_info in enumerate(sample_crops):
            row = i // cols
            col = i % cols
            
            # Load crop image
            crop_path = output_dir / crop_info['filename']
            
            if crop_path.exists():
                crop_img = cv2.imread(str(crop_path))
                crop_rgb = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
                
                axes[row, col].imshow(crop_rgb)
                
                # Title with info
                title = f"Crop {crop_info['crop_index']:03d}\n"
                title += f"{crop_info['crop_size']}×{crop_info['crop_size']}px\n"
                title += f"GPS: {crop_info['estimated_gps'][0]:.5f}, {crop_info['estimated_gps'][1]:.5f}"
                
                axes[row, col].set_title(title, fontsize=10)
                axes[row, col].axis('off')
            else:
                axes[row, col].text(0.5, 0.5, 'File not found', ha='center', va='center',
                                   transform=axes[row, col].transAxes)
                axes[row, col].set_title(f"Crop {i}")
        
        # Hide empty subplots
        for i in range(len(sample_crops), rows * cols):
            row = i // cols
            col = i % cols
            axes[row, col].axis('off')
        
        plt.tight_layout()
        plt.suptitle(f'Extracted Drone Training Crops from {os.path.basename(self.tif_path)}', 
                     fontsize=16, y=1.02)
        plt.show()

# Usage functions
def analyze_ukraine_tif():
    """Analyze your Ukraine TIF file"""
    
    print("🇺🇦" + "="*60 + "🇺🇦")
    print("    ANALYZING UKRAINE SATELLITE TIF FILE")
    print("🇺🇦" + "="*60 + "🇺🇦")
    
    # Initialize extractor
    extractor = TIFDroneExtractor("sentinel2_ukraine_10km.TIF")
    
    if extractor.image is not None:
        # Display overview
        extractor.display_overview()
        
        # Extract drone training crops
        crops = extractor.extract_drone_training_crops(
            num_crops=50,                    # Number of crops to extract
            crop_sizes=[128, 256, 512]      # Different sizes for variety
        )
        
        # Visualize extracted crops
        extractor.visualize_extracted_crops(num_display=12)
        
        print(f"\n🎉 ANALYSIS COMPLETE!")
        print(f"   Original image: {extractor.image.shape}")
        print(f"   Extracted crops: {len(crops)}")
        print(f"   Ready for RL training!")
        
        return extractor, crops
    else:
        print("❌ Could not load TIF file")
        return None, []

if __name__ == "__main__":
    print("📁 TIF FILE DRONE EXTRACTOR")
    print("="*50)
    print()
    print("🎯 WHAT IT DOES:")
    print("   • Loads your existing sentinel2_ukraine_10km.TIF")
    print("   • Analyzes image properties and displays overview")
    print("   • Extracts realistic drone-like crops for training")
    print("   • Applies transformations (rotation, brightness, noise)")
    print("   • Estimates GPS coordinates for each crop")
    print("   • Creates training dataset from your existing file")
    print()
    print("🚀 USAGE:")
    print("   extractor, crops = analyze_ukraine_tif()")
    print()
    print("📊 OUTPUT:")
    print("   • Image analysis and visualization")
    print("   • 50 drone training crops (128px, 256px, 512px)")
    print("   • GPS coordinates for each crop")
    print("   • Metadata for RL training")
    print()
    print("🎉 Perfect for creating RL training data from your existing file!")
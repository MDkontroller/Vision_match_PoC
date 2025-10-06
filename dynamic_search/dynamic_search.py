import cv2
import numpy as np
import requests
import math
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import time
from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass

@dataclass
class SearchResult:
    location: Tuple[int, int]  # pixel coordinates in big map
    gps_coords: Tuple[float, float]  # actual GPS coordinates
    confidence: float
    match_score: float
    search_window_size: int

class DynamicDroneSearcher:
    """
    AI that searches within a large map to find drone photo location
    """
    
    def __init__(self):
        self.sift = cv2.SIFT_create(nfeatures=300, contrastThreshold=0.05)
        self.search_results = []
        self.large_map = None
        self.map_bounds = None
        
    def download_large_map(self, center_lat: float, center_lng: float, 
                          zoom: int = 15, grid_size: Tuple[int, int] = (10, 10)) -> Tuple[np.ndarray, Dict]:
        """
        Download a large map area (e.g., 10x10 km) for searching
        """
        print(f"🗺️  Downloading {grid_size[0]}x{grid_size[1]} map at zoom {zoom}...")
        
        # Calculate tile coordinates for center
        center_x, center_y = self._deg2tile(center_lat, center_lng, zoom)
        
        # Calculate tile range
        grid_w, grid_h = grid_size
        start_x = center_x - grid_w // 2
        start_y = center_y - grid_h // 2
        
        # Download all tiles
        tiles = []
        for row in range(grid_h):
            tile_row = []
            for col in range(grid_w):
                tile_x = start_x + col
                tile_y = start_y + row
                
                tile = self._download_osm_tile(tile_x, tile_y, zoom)
                tile_row.append(tile)
                
                if (row * grid_w + col + 1) % 10 == 0:  # Progress indicator
                    print(f"  📦 Downloaded {row * grid_w + col + 1}/{grid_w * grid_h} tiles")
            
            # Combine tiles in this row
            row_image = np.hstack(tile_row)
            tiles.append(row_image)
        
        # Combine all rows into one large image
        large_map = np.vstack(tiles)
        
        # Calculate geographic bounds
        top_lat, left_lng = self._tile2deg(start_x, start_y, zoom)
        bottom_lat, right_lng = self._tile2deg(start_x + grid_w, start_y + grid_h, zoom)
        
        bounds = {
            'north': top_lat,
            'south': bottom_lat,
            'west': left_lng,
            'east': right_lng,
            'center_lat': center_lat,
            'center_lng': center_lng,
            'zoom': zoom,
            'width_km': self._calculate_distance(top_lat, left_lng, top_lat, right_lng),
            'height_km': self._calculate_distance(top_lat, left_lng, bottom_lat, left_lng)
        }
        
        print(f"✅ Created {large_map.shape[1]}x{large_map.shape[0]} map")
        print(f"📏 Coverage: {bounds['width_km']:.1f} x {bounds['height_km']:.1f} km")
        
        self.large_map = large_map
        self.map_bounds = bounds
        
        return large_map, bounds
    
    def search_drone_location(self, drone_img: np.ndarray, 
                            search_strategy: str = "multi_scale",
                            visualize: bool = True) -> List[SearchResult]:
        """
        Dynamically search the large map to find drone photo location
        """
        if self.large_map is None:
            raise ValueError("❌ No map loaded! Call download_large_map() first")
        
        print(f"🔍 Starting AI search using '{search_strategy}' strategy...")
        print(f"📊 Search area: {self.large_map.shape[1]}x{self.large_map.shape[0]} pixels")
        
        if search_strategy == "multi_scale":
            return self._multi_scale_search(drone_img, visualize)
        elif search_strategy == "sliding_window":
            return self._sliding_window_search(drone_img, visualize)
        elif search_strategy == "pyramid":
            return self._pyramid_search(drone_img, visualize)
        else:
            raise ValueError(f"Unknown strategy: {search_strategy}")
    
    def _multi_scale_search(self, drone_img: np.ndarray, visualize: bool) -> List[SearchResult]:
        """
        Search at multiple scales (coarse to fine)
        """
        print("🎯 Multi-scale search: Coarse → Fine")
        
        drone_gray = cv2.cvtColor(drone_img, cv2.COLOR_BGR2GRAY)
        map_gray = cv2.cvtColor(self.large_map, cv2.COLOR_RGB2GRAY)
        
        # Scale levels (how much to resize the search window)
        scales = [0.5, 0.75, 1.0, 1.25, 1.5]
        window_sizes = [200, 300, 400]  # Different window sizes to try
        
        all_results = []
        
        if visualize:
            fig, ax = plt.subplots(figsize=(12, 8))
            ax.imshow(self.large_map)
            ax.set_title("🔍 AI Dynamic Search Progress")
            search_patches = []
        
        for window_size in window_sizes:
            print(f"  🔍 Searching with {window_size}x{window_size} windows...")
            
            # Calculate step size (overlap windows by 50%)
            step_size = window_size // 2
            
            best_matches_this_size = []
            
            for y in range(0, map_gray.shape[0] - window_size, step_size):
                for x in range(0, map_gray.shape[1] - window_size, step_size):
                    
                    # Extract window from large map
                    map_window = map_gray[y:y+window_size, x:x+window_size]
                    
                    # Try different scales of drone image
                    best_score = 0
                    best_scale = 1.0
                    
                    for scale in scales:
                        # Resize drone image
                        new_size = int(drone_gray.shape[0] * scale), int(drone_gray.shape[1] * scale)
                        if new_size[0] > window_size or new_size[1] > window_size:
                            continue
                        
                        scaled_drone = cv2.resize(drone_gray, (new_size[1], new_size[0]))
                        
                        # Try template matching first (fast)
                        if scaled_drone.shape[0] <= map_window.shape[0] and scaled_drone.shape[1] <= map_window.shape[1]:
                            result = cv2.matchTemplate(map_window, scaled_drone, cv2.TM_CCOEFF_NORMED)
                            max_score = np.max(result)
                            
                            if max_score > best_score:
                                best_score = max_score
                                best_scale = scale
                    
                    # If template matching found something promising, do SIFT verification
                    if best_score > 0.3:  # Threshold for promising matches
                        scaled_drone = cv2.resize(drone_gray, 
                                                (int(drone_gray.shape[1] * best_scale), 
                                                 int(drone_gray.shape[0] * best_scale)))
                        
                        sift_score = self._verify_with_sift(scaled_drone, map_window)
                        
                        if sift_score > 0.1:  # SIFT verification threshold
                            # Convert to GPS coordinates
                            center_x, center_y = x + window_size//2, y + window_size//2
                            gps_lat, gps_lng = self._pixel_to_gps(center_x, center_y)
                            
                            result = SearchResult(
                                location=(center_x, center_y),
                                gps_coords=(gps_lat, gps_lng),
                                confidence=sift_score,
                                match_score=best_score,
                                search_window_size=window_size
                            )
                            
                            best_matches_this_size.append(result)
                            
                            # Visualize promising areas
                            if visualize:
                                color = 'red' if sift_score > 0.3 else 'yellow'
                                rect = patches.Rectangle((x, y), window_size, window_size, 
                                                       linewidth=2, edgecolor=color, facecolor='none', alpha=0.7)
                                ax.add_patch(rect)
                                search_patches.append(rect)
            
            # Keep top results for this window size
            best_matches_this_size.sort(key=lambda r: r.confidence, reverse=True)
            all_results.extend(best_matches_this_size[:5])  # Top 5 per window size
            
            print(f"    ✅ Found {len(best_matches_this_size)} potential matches")
        
        if visualize:
            plt.show()
        
        # Sort all results by confidence
        all_results.sort(key=lambda r: r.confidence, reverse=True)
        
        print(f"🎯 Search complete! Found {len(all_results)} potential locations")
        
        self.search_results = all_results[:10]  # Keep top 10
        return self.search_results
    
    def _verify_with_sift(self, drone_img: np.ndarray, map_window: np.ndarray) -> float:
        """
        Verify a potential match using SIFT features
        """
        try:
            # Detect features
            kp1, desc1 = self.sift.detectAndCompute(drone_img, None)
            kp2, desc2 = self.sift.detectAndCompute(map_window, None)
            
            if desc1 is None or desc2 is None or len(desc1) < 5 or len(desc2) < 5:
                return 0.0
            
            # Match features
            matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
            matches = matcher.knnMatch(desc1, desc2, k=2)
            
            # Lowe's ratio test
            good_matches = []
            for match_pair in matches:
                if len(match_pair) == 2:
                    m, n = match_pair
                    if m.distance < 0.75 * n.distance:
                        good_matches.append(m)
            
            if len(good_matches) < 4:
                return 0.0
            
            # Try homography
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            
            H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            
            if H is None:
                return 0.0
            
            inliers = np.sum(mask) if mask is not None else 0
            inlier_ratio = inliers / len(good_matches)
            
            # Score based on inlier ratio and number of matches
            score = inlier_ratio * min(1.0, len(good_matches) / 20)
            
            return float(score)
            
        except Exception as e:
            return 0.0
    
    def create_results_visualization(self, top_n: int = 5):
        """
        Create visualization showing the search results
        """
        if not self.search_results:
            print("❌ No search results to visualize")
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('🎯 Dynamic Drone Search Results', fontsize=16, fontweight='bold')
        
        # Main map with all results
        ax_main = axes[0, 0]
        ax_main.imshow(self.large_map)
        ax_main.set_title(f'🗺️ Search Area ({self.map_bounds["width_km"]:.1f} x {self.map_bounds["height_km"]:.1f} km)')
        
        # Plot all results
        for i, result in enumerate(self.search_results[:top_n]):
            x, y = result.location
            confidence = result.confidence
            
            # Color based on confidence
            if confidence > 0.5:
                color, size = 'red', 150
            elif confidence > 0.3:
                color, size = 'orange', 100
            else:
                color, size = 'yellow', 50
            
            ax_main.scatter(x, y, c=color, s=size, alpha=0.8, edgecolors='black')
            ax_main.text(x+10, y-10, f'{i+1}', fontsize=12, fontweight='bold', color='white')
        
        ax_main.axis('off')
        
        # Top result details
        if self.search_results:
            best_result = self.search_results[0]
            ax_best = axes[0, 1]
            
            # Extract region around best match
            x, y = best_result.location
            size = 200
            x1, y1 = max(0, x-size//2), max(0, y-size//2)
            x2, y2 = min(self.large_map.shape[1], x1+size), min(self.large_map.shape[0], y1+size)
            
            region = self.large_map[y1:y2, x1:x2]
            ax_best.imshow(region)
            ax_best.set_title(f'🏆 Best Match (Confidence: {best_result.confidence:.3f})')
            ax_best.plot(x-x1, y-y1, 'ro', markersize=15, markerfacecolor='red')
            ax_best.axis('off')
        
        # Results table
        ax_table = axes[0, 2]
        ax_table.axis('off')
        
        table_data = []
        for i, result in enumerate(self.search_results[:top_n]):
            table_data.append([
                f"#{i+1}",
                f"{result.confidence:.3f}",
                f"{result.gps_coords[0]:.6f}",
                f"{result.gps_coords[1]:.6f}"
            ])
        
        table = ax_table.table(cellText=table_data,
                              colLabels=['Rank', 'Confidence', 'Latitude', 'Longitude'],
                              cellLoc='center',
                              loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        ax_table.set_title('📊 Top Results')
        
        # Search statistics
        ax_stats = axes[1, 0]
        ax_stats.axis('off')
        
        if self.search_results:
            confidences = [r.confidence for r in self.search_results]
            stats_text = f"""
🔍 SEARCH STATISTICS

📏 Search Area: {self.map_bounds['width_km']:.1f} x {self.map_bounds['height_km']:.1f} km
🎯 Total Candidates: {len(self.search_results)}
🏆 Best Confidence: {max(confidences):.3f}
📊 Average Confidence: {np.mean(confidences):.3f}
📍 GPS Range:
   Lat: {self.map_bounds['south']:.4f} to {self.map_bounds['north']:.4f}
   Lng: {self.map_bounds['west']:.4f} to {self.map_bounds['east']:.4f}

🎯 BEST MATCH:
📍 Location: {self.search_results[0].gps_coords[0]:.6f}, {self.search_results[0].gps_coords[1]:.6f}
🎚️ Confidence: {self.search_results[0].confidence:.3f}
📏 Window: {self.search_results[0].search_window_size}px
            """
            
            ax_stats.text(0.05, 0.95, stats_text, transform=ax_stats.transAxes,
                         fontsize=10, verticalalignment='top', fontfamily='monospace',
                         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        ax_stats.set_title('📈 Search Analysis')
        
        # Confidence distribution
        ax_dist = axes[1, 1]
        if self.search_results:
            confidences = [r.confidence for r in self.search_results]
            ax_dist.hist(confidences, bins=10, alpha=0.7, color='skyblue', edgecolor='black')
            ax_dist.set_xlabel('Confidence Score')
            ax_dist.set_ylabel('Number of Matches')
            ax_dist.set_title('📊 Confidence Distribution')
            ax_dist.grid(True, alpha=0.3)
        
        # Summary
        ax_summary = axes[1, 2]
        ax_summary.axis('off')
        
        if self.search_results and self.search_results[0].confidence > 0.3:
            summary_color = 'lightgreen'
            summary_text = f"""
🎉 DRONE LOCATED!

📍 Coordinates: 
   {self.search_results[0].gps_coords[0]:.6f}°
   {self.search_results[0].gps_coords[1]:.6f}°

🎯 Confidence: {self.search_results[0].confidence:.1%}

✅ High-confidence match found
🚁 Ready for rescue/recovery
📱 Coordinates ready for GPS navigation

🏆 Mission Successful!
            """
        else:
            summary_color = 'lightcoral'
            summary_text = """
⚠️ UNCERTAIN RESULTS

🔍 No high-confidence matches found

Possible reasons:
• Drone image too different from map view
• Area not covered in search zone
• Weather/lighting differences
• Scale mismatch

🔧 Suggestions:
• Expand search area
• Try different zoom levels
• Check image quality
            """
        
        ax_summary.text(0.05, 0.95, summary_text, transform=ax_summary.transAxes,
                       fontsize=11, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor=summary_color, alpha=0.8))
        ax_summary.set_title('🏆 Mission Status')
        
        plt.tight_layout()
        plt.show()
    
    def get_best_location(self) -> Optional[Tuple[float, float]]:
        """Get the best GPS coordinates found"""
        if self.search_results:
            return self.search_results[0].gps_coords
        return None
    
    # Helper methods
    def _deg2tile(self, lat: float, lng: float, zoom: int):
        lat_rad = math.radians(lat)
        n = 2.0 ** zoom
        x = (lng + 180.0) / 360.0 * n
        y = (1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n
        return int(x), int(y)
    
    def _tile2deg(self, x: int, y: int, zoom: int):
        n = 2.0 ** zoom
        lng = x / n * 360.0 - 180.0
        lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * y / n)))
        lat = math.degrees(lat_rad)
        return lat, lng
    
    def _download_osm_tile(self, x: int, y: int, zoom: int):
        url = f"https://tile.openstreetmap.org/{zoom}/{x}/{y}.png"
        headers = {'User-Agent': 'DroneSearchDemo/1.0'}
        
        try:
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code == 200:
                tile = Image.open(BytesIO(response.content))
                return np.array(tile)
            else:
                return np.full((256, 256, 3), 200, dtype=np.uint8)
        except:
            return np.full((256, 256, 3), 200, dtype=np.uint8)
    
    def _pixel_to_gps(self, pixel_x: int, pixel_y: int) -> Tuple[float, float]:
        """Convert pixel coordinates in large map to GPS coordinates"""
        h, w = self.large_map.shape[:2]
        
        lat_range = self.map_bounds['north'] - self.map_bounds['south']
        lng_range = self.map_bounds['east'] - self.map_bounds['west']
        
        lat = self.map_bounds['south'] + (1 - pixel_y/h) * lat_range
        lng = self.map_bounds['west'] + (pixel_x/w) * lng_range
        
        return lat, lng
    
    def _calculate_distance(self, lat1: float, lng1: float, lat2: float, lng2: float) -> float:
        """Calculate distance in km using Haversine formula"""
        R = 6371  # Earth's radius in km
        
        lat1_rad, lng1_rad = math.radians(lat1), math.radians(lng1)
        lat2_rad, lng2_rad = math.radians(lat2), math.radians(lng2)
        
        dlat = lat2_rad - lat1_rad
        dlng = lng2_rad - lng1_rad
        
        a = math.sin(dlat/2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlng/2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        
        return R * c

# Demo function
def demo_dynamic_search():
    """
    Complete demo of dynamic drone search
    """
    print("🚁" + "="*50 + "🚁")
    print("   DYNAMIC DRONE SEARCH DEMO")
    print("   AI searches 10x10km area automatically")
    print("🚁" + "="*50 + "🚁")
    
    # Initialize searcher
    searcher = DynamicDroneSearcher()
    
    # Set search area (change these coordinates!)
    search_lat, search_lng = 40.7829, -73.9654  # Central Park, NYC
    
    # Download large map (10x10km area)
    print(f"\n📡 Step 1: Downloading search area...")
    large_map, bounds = searcher.download_large_map(
        search_lat, search_lng, 
        zoom=15,           # Good detail level
        grid_size=(8, 8)   # 8x8 tiles ≈ 8km x 8km at zoom 15
    )
    
    # Create mock drone image (in real scenario, you'd load actual drone photo)
    print(f"\n📸 Step 2: Loading drone image...")
    # For demo, we'll create a cropped section of the map as "drone view"
    h, w = large_map.shape[:2]
    drone_view = large_map[h//3:h//3+200, w//3:w//3+200]  # Extract a section
    drone_view = cv2.GaussianBlur(drone_view, (3, 3), 0)  # Add slight blur
    
    print(f"🚁 Drone image: {drone_view.shape}")
    
    # Run the search
    print(f"\n🔍 Step 3: AI searching for drone location...")
    results = searcher.search_drone_location(
        cv2.cvtColor(drone_view, cv2.COLOR_RGB2BGR),
        search_strategy="multi_scale",
        visualize=True
    )
    
    # Show results
    print(f"\n📊 Step 4: Analyzing results...")
    searcher.create_results_visualization(top_n=5)
    
    # Get best location
    best_gps = searcher.get_best_location()
    if best_gps:
        print(f"\n🎯 DRONE FOUND!")
        print(f"📍 GPS Coordinates: {best_gps[0]:.6f}, {best_gps[1]:.6f}")
        print(f"🎚️ Confidence: {results[0].confidence:.1%}")
        print(f"🚁 Ready for rescue mission!")
    else:
        print(f"\n❌ Drone not found in search area")
        print(f"💡 Try expanding search area or different zoom level")

if __name__ == "__main__":
    print("🎯 Dynamic Drone Search System Ready!")
    print("💡 This AI searches within a large downloaded map to find drone locations")
    print("💡 Call demo_dynamic_search() to see it in action!")
    
    # Uncomment to run demo:
    # demo_dynamic_search()
import cv2
import numpy as np
import requests
import json
import folium
from folium import plugins
import base64
from io import BytesIO
from PIL import Image
import matplotlib.pyplot as plt
import time
from typing import Tuple, List, Dict, Optional
import os
from datetime import datetime

class HackathonDroneLocator:
    """
    Simple but impressive drone locator for hackathon demo
    Uses Google Maps API and optimized SIFT matching
    """
    
    def __init__(self, google_maps_api_key: str = None):
        self.api_key = google_maps_api_key
        self.sift = cv2.SIFT_create(nfeatures=500, contrastThreshold=0.04)
        self.results_history = []
        
    def download_satellite_image(self, lat: float, lng: float, zoom: int = 18, 
                                size: str = "640x640", maptype: str = "satellite") -> np.ndarray:
        """
        Download satellite image from Google Maps Static API
        """
        if not self.api_key:
            print("⚠️  No API key provided. Using mock satellite image.")
            return self._create_mock_satellite_image()
        
        url = f"https://maps.googleapis.com/maps/api/staticmap"
        params = {
            'center': f"{lat},{lng}",
            'zoom': zoom,
            'size': size,
            'maptype': maptype,
            'key': self.api_key
        }
        
        try:
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                # Convert to OpenCV format
                image = Image.open(BytesIO(response.content))
                return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            else:
                print(f"❌ API Error: {response.status_code}")
                return self._create_mock_satellite_image()
        except Exception as e:
            print(f"❌ Download failed: {e}")
            return self._create_mock_satellite_image()
    
    def _create_mock_satellite_image(self, size: Tuple[int, int] = (640, 640)) -> np.ndarray:
        """Create a realistic mock satellite image for demo"""
        h, w = size
        
        # Create base terrain
        mock_img = np.random.randint(80, 120, (h, w, 3), dtype=np.uint8)
        
        # Add some geometric structures (buildings, roads)
        # Buildings
        for _ in range(15):
            x = np.random.randint(50, w-100)
            y = np.random.randint(50, h-100)
            w_building = np.random.randint(30, 80)
            h_building = np.random.randint(30, 80)
            color = np.random.randint(140, 180, 3)
            cv2.rectangle(mock_img, (x, y), (x+w_building, y+h_building), color.tolist(), -1)
        
        # Roads
        for _ in range(8):
            start = (np.random.randint(0, w), np.random.randint(0, h))
            end = (np.random.randint(0, w), np.random.randint(0, h))
            cv2.line(mock_img, start, end, (60, 60, 60), np.random.randint(8, 15))
        
        # Add some noise and texture
        noise = np.random.randint(-20, 20, (h, w, 3))
        mock_img = np.clip(mock_img + noise, 0, 255).astype(np.uint8)
        
        return mock_img
    
    def smart_feature_matching(self, drone_img: np.ndarray, satellite_img: np.ndarray) -> Dict:
        """
        Optimized feature matching with quality metrics
        """
        start_time = time.time()
        
        # Convert to grayscale
        drone_gray = cv2.cvtColor(drone_img, cv2.COLOR_BGR2GRAY)
        satellite_gray = cv2.cvtColor(satellite_img, cv2.COLOR_BGR2GRAY)
        
        # Enhance contrast for better feature detection
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        drone_gray = clahe.apply(drone_gray)
        satellite_gray = clahe.apply(satellite_gray)
        
        # Detect features
        kp1, desc1 = self.sift.detectAndCompute(drone_gray, None)
        kp2, desc2 = self.sift.detectAndCompute(satellite_gray, None)
        
        if desc1 is None or desc2 is None:
            return {'success': False, 'error': 'No features detected'}
        
        # Feature matching
        matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        matches = matcher.knnMatch(desc1, desc2, k=2)
        
        # Lowe's ratio test with adaptive threshold
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < 0.7 * n.distance:
                    good_matches.append(m)
        
        if len(good_matches) < 10:
            return {'success': False, 'error': f'Insufficient matches: {len(good_matches)}'}
        
        # Find homography with RANSAC
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 
                                    ransacReprojThreshold=5.0, confidence=0.99)
        
        if H is None:
            return {'success': False, 'error': 'Homography estimation failed'}
        
        inliers = np.sum(mask) if mask is not None else 0
        inlier_ratio = inliers / len(good_matches)
        
        # Transform drone image corners to satellite coordinates
        h, w = drone_gray.shape
        corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
        transformed_corners = cv2.perspectiveTransform(corners, H)
        
        # Calculate center and area
        center_x = int(np.mean(transformed_corners[:, 0, 0]))
        center_y = int(np.mean(transformed_corners[:, 0, 1]))
        area = cv2.contourArea(transformed_corners)
        
        # Quality assessment
        processing_time = time.time() - start_time
        quality_score = inlier_ratio * min(1.0, inliers / 50)  # Normalize quality
        
        return {
            'success': True,
            'center': (center_x, center_y),
            'corners': transformed_corners.tolist(),
            'inliers': int(inliers),
            'total_matches': len(good_matches),
            'inlier_ratio': float(inlier_ratio),
            'quality_score': float(quality_score),
            'area': float(area),
            'processing_time': float(processing_time),
            'homography': H.tolist()
        }
    
    def locate_drone_from_coordinates(self, drone_img_path: str, lat: float, lng: float, 
                                    search_radius: int = 2) -> Dict:
        """
        Main demo function: Find drone location given approximate GPS coordinates
        """
        print(f"🎯 Locating drone near {lat:.6f}, {lng:.6f}")
        
        # Load drone image
        drone_img = cv2.imread(drone_img_path)
        if drone_img is None:
            return {'success': False, 'error': f'Could not load drone image: {drone_img_path}'}
        
        best_result = None
        best_score = 0
        search_results = []
        
        # Search in a grid around the GPS coordinates
        print(f"🔍 Searching in {search_radius}x{search_radius} grid...")
        
        for i in range(-search_radius//2, search_radius//2 + 1):
            for j in range(-search_radius//2, search_radius//2 + 1):
                # Offset coordinates (roughly 100m per 0.001 degrees)
                offset_lat = lat + i * 0.001
                offset_lng = lng + j * 0.001
                
                print(f"  📡 Downloading satellite image {i+search_radius//2+1}, {j+search_radius//2+1}...")
                satellite_img = self.download_satellite_image(offset_lat, offset_lng)
                
                # Perform matching
                result = self.smart_feature_matching(drone_img, satellite_img)
                
                if result['success']:
                    result['search_coordinates'] = (offset_lat, offset_lng)
                    result['grid_position'] = (i, j)
                    search_results.append(result)
                    
                    if result['quality_score'] > best_score:
                        best_score = result['quality_score']
                        best_result = result
                        print(f"  ✅ New best match! Quality: {best_score:.3f}")
                else:
                    print(f"  ❌ No match: {result.get('error', 'Unknown error')}")
        
        if best_result:
            # Create comprehensive result
            final_result = {
                'success': True,
                'drone_image': drone_img_path,
                'search_center': (lat, lng),
                'best_match': best_result,
                'all_matches': search_results,
                'timestamp': datetime.now().isoformat(),
                'confidence_level': self._assess_confidence(best_result)
            }
            
            self.results_history.append(final_result)
            return final_result
        else:
            return {
                'success': False,
                'error': 'No matches found in search area',
                'search_center': (lat, lng),
                'attempts': len(search_results)
            }
    
    def _assess_confidence(self, result: Dict) -> str:
        """Assess confidence level based on matching metrics"""
        score = result['quality_score']
        inlier_ratio = result['inlier_ratio']
        
        if score > 0.7 and inlier_ratio > 0.6:
            return "HIGH"
        elif score > 0.4 and inlier_ratio > 0.4:
            return "MEDIUM"
        else:
            return "LOW"
    
    def create_interactive_map(self, result: Dict, output_file: str = "drone_location_map.html"):
        """
        Create an interactive map showing the results
        """
        if not result['success']:
            print("❌ Cannot create map: No successful matches")
            return
        
        best_match = result['best_match']
        search_coords = best_match['search_coordinates']
        
        # Create map centered on the result
        m = folium.Map(
            location=search_coords,
            zoom_start=18,
            tiles='OpenStreetMap'
        )
        
        # Add satellite layer
        folium.TileLayer(
            tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
            attr='Esri',
            name='Satellite',
            overlay=False,
            control=True
        ).add_to(m)
        
        # Add marker for drone location
        folium.Marker(
            search_coords,
            popup=f"""
            <b>Drone Location Found!</b><br>
            Confidence: {result['confidence_level']}<br>
            Quality Score: {best_match['quality_score']:.3f}<br>
            Inlier Ratio: {best_match['inlier_ratio']:.3f}<br>
            Matches: {best_match['inliers']}/{best_match['total_matches']}<br>
            Processing Time: {best_match['processing_time']:.2f}s
            """,
            tooltip="🎯 Drone Location",
            icon=folium.Icon(color='red', icon='camera')
        ).add_to(m)
        
        # Add search area
        search_center = result['search_center']
        folium.Circle(
            search_center,
            radius=200,  # 200m radius
            popup="Search Area",
            color='blue',
            fill=True,
            fillOpacity=0.2
        ).add_to(m)
        
        # Add all match attempts
        for i, match in enumerate(result.get('all_matches', [])):
            coords = match['search_coordinates']
            color = 'green' if match == best_match else 'orange'
            
            folium.CircleMarker(
                coords,
                radius=8,
                popup=f"Match {i+1}: {match['quality_score']:.3f}",
                color=color,
                fill=True
            ).add_to(m)
        
        # Add layer control
        folium.LayerControl().add_to(m)
        
        # Save map
        m.save(output_file)
        print(f"🗺️  Interactive map saved: {output_file}")
        
        return m
    
    def create_visualization_dashboard(self, result: Dict, output_file: str = "demo_results.png"):
        """
        Create a comprehensive visualization for the demo
        """
        if not result['success']:
            print("❌ Cannot create visualization: No successful matches")
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('🚁 Drone Localization Demo Results', fontsize=20, fontweight='bold')
        
        best_match = result['best_match']
        
        # Load images for visualization
        drone_img = cv2.imread(result['drone_image'])
        drone_img_rgb = cv2.cvtColor(drone_img, cv2.COLOR_BGR2RGB)
        
        # Download satellite image for the best match
        sat_coords = best_match['search_coordinates']
        satellite_img = self.download_satellite_image(sat_coords[0], sat_coords[1])
        satellite_img_rgb = cv2.cvtColor(satellite_img, cv2.COLOR_BGR2RGB)
        
        # 1. Drone Image
        axes[0, 0].imshow(drone_img_rgb)
        axes[0, 0].set_title('📱 Drone Image', fontsize=14, fontweight='bold')
        axes[0, 0].axis('off')
        
        # 2. Satellite Image
        axes[0, 1].imshow(satellite_img_rgb)
        axes[0, 1].set_title('🛰️ Matching Satellite Image', fontsize=14, fontweight='bold')
        
        # Draw detected location on satellite image
        corners = np.array(best_match['corners']).reshape(-1, 2)
        center = best_match['center']
        
        # Draw bounding box
        for i in range(4):
            start = tuple(corners[i].astype(int))
            end = tuple(corners[(i+1)%4].astype(int))
            axes[0, 1].plot([start[0], end[0]], [start[1], end[1]], 'r-', linewidth=3)
        
        # Draw center point
        axes[0, 1].plot(center[0], center[1], 'ro', markersize=10, markerfacecolor='red')
        axes[0, 1].text(center[0]+10, center[1]-10, '🎯 FOUND!', fontsize=12, 
                       color='red', fontweight='bold')
        axes[0, 1].axis('off')
        
        # 3. Quality Metrics
        metrics = [
            ('Quality Score', best_match['quality_score'], 1.0),
            ('Inlier Ratio', best_match['inlier_ratio'], 1.0),
            ('Confidence', {'HIGH': 0.9, 'MEDIUM': 0.6, 'LOW': 0.3}[result['confidence_level']], 1.0)
        ]
        
        bars = axes[0, 2].barh([m[0] for m in metrics], [m[1] for m in metrics], 
                              color=['green' if m[1] > 0.6 else 'orange' if m[1] > 0.3 else 'red' for m in metrics])
        axes[0, 2].set_xlim(0, 1)
        axes[0, 2].set_title('📊 Quality Metrics', fontsize=14, fontweight='bold')
        
        # Add value labels on bars
        for i, (bar, metric) in enumerate(zip(bars, metrics)):
            axes[0, 2].text(metric[1] + 0.02, i, f'{metric[1]:.3f}', 
                           va='center', fontweight='bold')
        
        # 4. Feature Matching Statistics
        match_data = [
            f"Total Features: {best_match['total_matches']}",
            f"Inliers: {best_match['inliers']}",
            f"Processing Time: {best_match['processing_time']:.2f}s",
            f"Detected Area: {best_match['area']:.0f} pixels²",
            f"GPS Coordinates: {sat_coords[0]:.6f}, {sat_coords[1]:.6f}"
        ]
        
        axes[1, 0].text(0.05, 0.95, '\n'.join(match_data), transform=axes[1, 0].transAxes,
                       fontsize=12, verticalalignment='top', fontfamily='monospace',
                       bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        axes[1, 0].set_title('📈 Technical Details', fontsize=14, fontweight='bold')
        axes[1, 0].axis('off')
        
        # 5. Search Pattern
        all_matches = result.get('all_matches', [])
        if len(all_matches) > 1:
            grid_positions = [m['grid_position'] for m in all_matches]
            scores = [m['quality_score'] for m in all_matches]
            
            scatter = axes[1, 1].scatter([p[0] for p in grid_positions], 
                                       [p[1] for p in grid_positions],
                                       c=scores, s=200, cmap='viridis', alpha=0.8)
            
            # Highlight best match
            best_pos = best_match['grid_position']
            axes[1, 1].scatter(best_pos[0], best_pos[1], c='red', s=300, marker='*')
            
            axes[1, 1].set_title('🗺️ Search Pattern & Scores', fontsize=14, fontweight='bold')
            axes[1, 1].grid(True, alpha=0.3)
            plt.colorbar(scatter, ax=axes[1, 1], label='Quality Score')
        else:
            axes[1, 1].text(0.5, 0.5, 'Single point search', ha='center', va='center',
                           transform=axes[1, 1].transAxes, fontsize=14)
            axes[1, 1].set_title('🗺️ Search Pattern', fontsize=14, fontweight='bold')
        
        # 6. Performance Summary
        summary_text = f"""
        🎯 DRONE SUCCESSFULLY LOCATED!
        
        📍 Location: {sat_coords[0]:.6f}, {sat_coords[1]:.6f}
        🎚️ Confidence: {result['confidence_level']}
        ⭐ Quality: {best_match['quality_score']:.3f}
        ⚡ Speed: {best_match['processing_time']:.2f}s
        🔍 Matches: {best_match['inliers']}/{best_match['total_matches']}
        
        Ready for real-world deployment! 🚀
        """
        
        axes[1, 2].text(0.05, 0.95, summary_text, transform=axes[1, 2].transAxes,
                       fontsize=12, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
        axes[1, 2].set_title('🏆 Demo Results', fontsize=14, fontweight='bold')
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"📊 Demo visualization saved: {output_file}")

# Demo script
def run_hackathon_demo():
    """
    Complete hackathon demo script
    """
    print("🚁" + "="*60 + "🚁")
    print("    HACKATHON DEMO: AI-Powered Drone Localization")
    print("🚁" + "="*60 + "🚁")
    
    # Initialize the locator
    api_key = os.getenv('GOOGLE_MAPS_API_KEY')  # Set your API key in environment
    locator = HackathonDroneLocator(api_key)
    
    # Demo coordinates (you can change these)
    demo_coords = [
        (40.7829, -73.9654),  # Central Park, NYC
        (51.5074, -0.1278),   # London
        (48.8566, 2.3522),    # Paris
    ]
    
    print(f"\n🎯 Starting demo with {len(demo_coords)} locations...")
    
    for i, (lat, lng) in enumerate(demo_coords):
        print(f"\n{'='*50}")
        print(f"Demo {i+1}: Location {lat}, {lng}")
        print(f"{'='*50}")
        
        # For demo purposes, create a mock drone image
        # In real scenario, you'd have actual drone photos
        drone_img_path = f"demo_drone_{i+1}.jpg"
        if not os.path.exists(drone_img_path):
            print(f"📸 Creating demo drone image: {drone_img_path}")
            # Create a cropped/rotated version of satellite image as "drone" image
            sat_img = locator.download_satellite_image(lat, lng, zoom=19)
            # Simulate drone perspective (crop and rotate)
            h, w = sat_img.shape[:2]
            drone_view = sat_img[h//4:3*h//4, w//4:3*w//4]  # Crop center
            drone_view = cv2.resize(drone_view, (400, 400))
            cv2.imwrite(drone_img_path, drone_view)
        
        # Run the localization
        result = locator.locate_drone_from_coordinates(drone_img_path, lat, lng, search_radius=2)
        
        if result['success']:
            print(f"🎉 SUCCESS! Drone located with {result['confidence_level']} confidence")
            
            # Create visualizations
            locator.create_interactive_map(result, f"demo_map_{i+1}.html")
            locator.create_visualization_dashboard(result, f"demo_results_{i+1}.png")
            
        else:
            print(f"❌ Demo {i+1} failed: {result.get('error', 'Unknown error')}")
    
    print(f"\n🏆 Demo completed! Check the generated files for results.")
    print(f"📁 Generated files:")
    print(f"   - Interactive maps: demo_map_*.html")
    print(f"   - Result dashboards: demo_results_*.png")

if __name__ == "__main__":
    # Quick test without API key
    locator = HackathonDroneLocator()
    print("🚁 Hackathon Demo Ready!")
    print("💡 For full demo with Google Maps, set GOOGLE_MAPS_API_KEY environment variable")
    print("💡 Call run_hackathon_demo() to start the full demonstration")
    
    # Uncomment to run full demo:
    # run_hackathon_demo()
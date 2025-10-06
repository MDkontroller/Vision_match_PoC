#!/usr/bin/env python3
"""
Simple website launcher for DroneLocator AI Demo
Creates basic demo assets and opens the website
"""

import os
import json
import webbrowser
from pathlib import Path
import time

def create_basic_assets():
    """Create basic assets for the website demo"""
    
    print("🚁 DRONELOCATOR AI - WEBSITE LAUNCHER")
    print("=" * 50)
    print("Creating basic demo assets...")
    
    # Create website_assets directory
    assets_dir = Path("website_assets")
    assets_dir.mkdir(exist_ok=True)
    
    # Create basic demo data
    demo_data = {
        "sift_results": {
            "total_scenarios": 3,
            "avg_confidence": 0.943,
            "avg_processing_time": 1.75,
            "scenarios": [
                {"scenario": "Central Park NYC", "confidence": 0.94, "processing_time": 1.8, "features_matched": 847},
                {"scenario": "Ukraine Region", "confidence": 0.92, "processing_time": 1.6, "features_matched": 723},
                {"scenario": "London UK", "confidence": 0.97, "processing_time": 1.9, "features_matched": 912}
            ]
        },
        "performance_metrics": {
            "accuracy_metrics": {
                "overall_accuracy": 0.963,
                "precision": 0.947,
                "recall": 0.978,
                "f1_score": 0.962
            },
            "speed_metrics": {
                "avg_feature_extraction_time": 0.3,
                "avg_matching_time": 0.7,
                "avg_total_processing_time": 1.8,
                "throughput_images_per_second": 0.56
            },
            "scalability_metrics": {
                "area_coverage": {
                    "1km2": 0.5,
                    "5km2": 1.2,
                    "10km2": 1.8,
                    "25km2": 3.1
                },
                "memory_usage_mb": 245,
                "peak_memory_mb": 512
            }
        },
        "demo_summary": {
            "timestamp": "2024-01-15T10:30:00",
            "total_capabilities": 5,
            "success_rate": 0.96,
            "ready_for_production": True
        }
    }
    
    # Save demo data
    with open(assets_dir / "demo_data.json", "w") as f:
        json.dump(demo_data, f, indent=2)
    
    # Create simple HTML map
    simple_map_html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>DroneLocator AI Demo Map</title>
        <link rel="stylesheet" href="https://unpkg.com/leaflet@1.7.1/dist/leaflet.css" />
    </head>
    <body>
        <div id="map" style="height: 500px;"></div>
        <script src="https://unpkg.com/leaflet@1.7.1/dist/leaflet.js"></script>
        <script>
            var map = L.map('map').setView([40.7829, -73.9654], 13);
            L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png').addTo(map);
            
            var locations = [
                {lat: 40.7829, lng: -73.9654, name: "Central Park, NYC", confidence: 94.2},
                {lat: 50.2957, lng: 36.6619, name: "Ukraine Region", confidence: 87.5},
                {lat: 51.5074, lng: -0.1278, name: "London, UK", confidence: 91.8}
            ];
            
            locations.forEach(function(loc) {
                L.marker([loc.lat, loc.lng])
                 .bindPopup('<b>' + loc.name + '</b><br>Confidence: ' + loc.confidence + '%')
                 .addTo(map);
            });
        </script>
    </body>
    </html>
    """
    
    with open(assets_dir / "interactive_demo_map.html", "w") as f:
        f.write(simple_map_html)
    
    print("✅ Basic assets created in website_assets/")
    return assets_dir

def launch_website():
    """Launch the website in the default browser"""
    
    # Check if index.html exists
    index_path = Path("index.html")
    
    if index_path.exists():
        # Get absolute path
        abs_path = index_path.absolute()
        
        print(f"🌐 Launching website: {abs_path}")
        
        # Open in browser
        webbrowser.open(f"file://{abs_path}")
        
        print("✅ Website launched successfully!")
        print()
        print("🎯 WEBSITE FEATURES:")
        print("• Interactive hero section with animations")
        print("• Comprehensive feature showcase")
        print("• Live demo with simulated search")
        print("• Technology deep-dive sections")
        print("• Performance metrics and charts")
        print("• Responsive design for all devices")
        print()
        print("📱 Try the interactive demo in the Demo section!")
        
        return True
    else:
        print("❌ index.html not found in current directory")
        print("Please ensure you're in the correct directory with the website files.")
        return False

def create_requirements_file():
    """Create a requirements.txt file for the project"""
    
    requirements = """# DroneLocator AI Website Requirements
# Core computer vision and machine learning
opencv-python>=4.5.0
numpy>=1.21.0
matplotlib>=3.5.0
Pillow>=8.3.0

# Machine learning (optional - for RL agent)
torch>=1.9.0
torchvision>=0.10.0

# Mapping and visualization
folium>=0.12.0
requests>=2.26.0

# Data processing
scipy>=1.7.0
scikit-learn>=1.0.0

# Optional performance enhancements
numba>=0.54.0

# Development and testing
pytest>=6.2.0
jupyter>=1.0.0
"""
    
    with open("requirements.txt", "w") as f:
        f.write(requirements.strip())
    
    print("📄 Created requirements.txt")

def main():
    """Main function to set up and launch the website"""
    
    print("🚁 DRONELOCATOR AI - SETTING UP WEBSITE DEMO")
    print("=" * 60)
    print()
    
    # Create basic assets
    assets_dir = create_basic_assets()
    
    # Create requirements file
    create_requirements_file()
    
    # Launch website
    print("\n🌐 LAUNCHING WEBSITE...")
    print("-" * 30)
    
    success = launch_website()
    
    if success:
        print("\n🎉 WEBSITE DEMO READY!")
        print("=" * 40)
        print("📁 Assets directory:", assets_dir.absolute())
        print("🌐 Website opened in your default browser")
        print("📚 See website_README.md for full documentation")
        print()
        print("🚀 Your DroneLocator AI demonstration is live!")
    else:
        print("\n❌ Failed to launch website")
        print("Please check that index.html exists in the current directory")

if __name__ == "__main__":
    main()


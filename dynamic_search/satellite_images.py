import requests
import cv2
import numpy as np
from PIL import Image
from io import BytesIO
import base64

def download_sentinel_hub(lat=50.2957, lng=36.6619, client_id="your_client_id", client_secret="your_client_secret"):
    """Download from Sentinel Hub with your credentials"""
    
    # Get access token
    token_url = "https://services.sentinel-hub.com/oauth/token"
    token_data = {
        'grant_type': 'client_credentials',
        'client_id': client_id,
        'client_secret': client_secret
    }
    
    token_response = requests.post(token_url, data=token_data)
    if token_response.status_code != 200:
        print("Authentication failed")
        return None
    
    access_token = token_response.json()['access_token']
    
    # Calculate bbox for 5x5km
    km_to_deg = 1/111
    buffer = 2.5 * km_to_deg  # 2.5km radius = 5km total
    
    bbox = [lng - buffer, lat - buffer, lng + buffer, lat + buffer]
    
    # Process API request
    url = "https://services.sentinel-hub.com/api/v1/process"
    
    headers = {
        'Authorization': f'Bearer {access_token}',
        'Content-Type': 'application/json'
    }
    
    payload = {
        "input": {
            "bounds": {
                "properties": {
                    "crs": "http://www.opengis.net/def/crs/EPSG/0/4326"
                },
                "bbox": bbox
            },
            "data": [{
                "type": "sentinel-2-l2a",
                "dataFilter": {
                    "timeRange": {
                        "from": "2024-01-01T00:00:00Z",
                        "to": "2024-12-31T23:59:59Z"
                    },
                    "maxCloudCoverage": 20
                }
            }]
        },
        "output": {
            "width": 512,
            "height": 512,
            "responses": [{
                "identifier": "default",
                "format": {
                    "type": "image/jpeg"
                }
            }]
        },
        "evalscript": """
            //VERSION=3
            function setup() {
                return {
                    input: ["B02", "B03", "B04"],
                    output: { bands: 3 }
                };
            }
            
            function evaluatePixel(sample) {
                return [sample.B04, sample.B03, sample.B02];
            }
        """
    }
    
    response = requests.post(url, headers=headers, json=payload)
    
    if response.status_code == 200:
        image = Image.open(BytesIO(response.content))
        image_array = np.array(image)
        image_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
        
        filename = f"sentinel2_hub_{lat}_{lng}.jpg"
        cv2.imwrite(filename, image_bgr)
        print(f"Downloaded Sentinel-2: {filename}")
        return image_bgr
    else:
        print(f"Request failed: {response.status_code}")
        print(response.text)
        return None

# Usage - replace with your credentials
# download_sentinel_hub(client_id="your_id", client_secret="your_secret")
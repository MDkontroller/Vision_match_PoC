from sentinelhub import (
    SentinelHubRequest, DataCollection, BBox, bbox_to_dimensions, SHConfig, MimeType
)
import numpy as np
import cv2
import rasterio

# 1 — Auth -------------------------------------------------------------------
cfg = SHConfig()
cfg.sh_client_id     = "sh-29927974-7d1f-4f2e-9751-f59a8149a944"
cfg.sh_client_secret = "Zqrx0QU4ivjOl3TM77zsm0Hhib8QWivO"
cfg.sh_auth_base_url = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
cfg.sh_base_url = "https://creodias.sentinel-hub.com"

# 2 — Define a 5 km × 5 km bounding box in WGS-84 ----------------------------
bbox = BBox(
    [36.6268,         # min lon
     50.2733,         # min lat
     36.6970,         # max lon
     50.3181],        # max lat
    crs="EPSG:4326"
)

# 3 — Pick a target resolution for the “zoom level” you want -----------------
size = bbox_to_dimensions(bbox, resolution=10)   # (width, height)

evalscript = """
//VERSION=3
function setup() {
  return {
    input: ["B04", "B03", "B02"],
    output: { bands: 3 }
  }
}
function evaluatePixel(sample) {
  return [sample.B04, sample.B03, sample.B02]
}
"""

# 4 — Sentinel Hub request ----------------------------------------------------
req = SentinelHubRequest(
    evalscript       = evalscript,
    input_data       = [SentinelHubRequest.input_data(
                           data_collection   = DataCollection.SENTINEL2_L2A,
                           time_interval     = ("2025-05-15", "2025-05-24"),
                           mosaicking_order  = 'mostRecent')],
    responses        = [SentinelHubRequest.output_response("default", MimeType.TIFF)],
    bbox             = bbox,
    size             = size,
    config           = cfg
)

# 5 — Download and save the image --------------------------------------------
rgb = req.get_data()[0]  # numpy array H×W×3, dtype=uint16

with rasterio.open("s2_chip.tif", "w", driver="GTiff",
                   height=rgb.shape[0], width=rgb.shape[1],
                   count=3, dtype=rgb.dtype, crs="EPSG:4326",
                   transform=req.get_transform()) as dst:
    for i in range(3):
        dst.write(rgb[:, :, i], i + 1)

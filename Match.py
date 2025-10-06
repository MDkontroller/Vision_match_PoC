import cv2
import numpy as np
import matplotlib.pyplot as plt

def find_drone_location_on_satellite(drone_path, satellite_path):
    """
    Find where the drone image location appears on the satellite image
    """
    # Load images
    drone = cv2.imread(drone_path)
    satellite = cv2.imread(satellite_path)
    
    if drone is None:
        print(f"Error: Could not load {drone_path}")
        return None
    if satellite is None:
        print(f"Error: Could not load {satellite_path}")
        return None
    
    print(f"Drone image shape: {drone.shape}")
    print(f"Satellite image shape: {satellite.shape}")
    
    # Convert to grayscale
    drone_gray = cv2.cvtColor(drone, cv2.COLOR_BGR2GRAY)
    satellite_gray = cv2.cvtColor(satellite, cv2.COLOR_BGR2GRAY)
    
    # 1. Feature detection and description
    # Try SIFT first (better for this task), fall back to ORB
    try:
        detector = cv2.SIFT_create(nfeatures=8000)
        print("Using SIFT detector")
    except:
        detector = cv2.ORB_create(nfeatures=8000)
        print("Using ORB detector")
    
    # Find keypoints and descriptors
    kp1, desc1 = detector.detectAndCompute(drone_gray, None)
    kp2, desc2 = detector.detectAndCompute(satellite_gray, None)
    
    if desc1 is None or desc2 is None:
        print("Error: Could not find features in one or both images")
        return None
    
    print(f"Found {len(kp1)} keypoints in drone image")
    print(f"Found {len(kp2)} keypoints in satellite image")
    
    # 2. Feature matching
    if detector.__class__.__name__ == 'SIFT':
        matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    else:
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    
    # Use knnMatch for ratio test
    matches = matcher.knnMatch(desc1, desc2, k=2)
    
    # Apply Lowe's ratio test
    good_matches = []
    for match_pair in matches:
        if len(match_pair) == 2:
            m, n = match_pair
            if m.distance < 0.7 * n.distance:  # Stricter ratio for better matches
                good_matches.append(m)
    
    print(f"Found {len(good_matches)} good matches")
    
    if len(good_matches) < 10:  # Need minimum matches
        print("Error: Not enough good matches found")
        return None
    
    # 3. Find the location using homography
    # Extract matched points
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    
    # Find homography
    H, mask = cv2.findHomography(src_pts, dst_pts, 
                                cv2.RANSAC, 
                                ransacReprojThreshold=5.0,
                                confidence=0.99)
    
    if H is None:
        print("Error: Could not compute homography")
        return None
    
    # Count inliers
    inliers = np.sum(mask)
    print(f"Homography found with {inliers} inliers out of {len(good_matches)} matches")
    
    # 4. Get the corners of the drone image and transform them
    h, w = drone_gray.shape
    corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
    
    # Transform corners to satellite image coordinates
    transformed_corners = cv2.perspectiveTransform(corners, H)
    
    # 5. Create visualization
    result_img = satellite.copy()
    
    # Draw the transformed region on satellite image
    pts = np.int32(transformed_corners).reshape((-1, 1, 2))
    cv2.polylines(result_img, [pts], True, (0, 255, 0), 3)
    
    # Calculate center of the region
    center_x = int(np.mean(transformed_corners[:, 0, 0]))
    center_y = int(np.mean(transformed_corners[:, 0, 1]))
    
    # Draw center point
    cv2.circle(result_img, (center_x, center_y), 10, (0, 0, 255), -1)
    cv2.putText(result_img, f"Drone Location: ({center_x}, {center_y})", 
                (center_x + 15, center_y - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # Save result
    cv2.imwrite("drone_location_found.jpg", result_img)
    
    # Optional: Create a detailed match visualization
    match_img = cv2.drawMatches(drone, kp1, satellite, kp2, 
                               [m for i, m in enumerate(good_matches) if mask[i]], 
                               None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.imwrite("feature_matches.jpg", match_img)
    
    print(f"Drone location center: ({center_x}, {center_y})")
    print("Results saved as:")
    print("- drone_location_found.jpg (satellite image with marked location)")
    print("- feature_matches.jpg (feature matching visualization)")
    
    return {
        'center': (center_x, center_y),
        'corners': transformed_corners,
        'homography': H,
        'inliers': inliers,
        'total_matches': len(good_matches)
    }

# Usage
if __name__ == "__main__":
    result = find_drone_location_on_satellite("drone.jpeg", "satellite.png")
    
    if result:
        print(f"\n✅ Success! Drone location found at: {result['center']}")
        print(f"Confidence: {result['inliers']}/{result['total_matches']} inlier matches")
    else:
        print("\n❌ Could not determine drone location")
        
        # Troubleshooting suggestions
        print("\nTroubleshooting suggestions:")
        print("1. Ensure both images have overlapping areas")
        print("2. Check image quality and lighting conditions")
        print("3. Try different feature detectors (SIFT vs ORB)")
        print("4. Adjust matching ratio threshold")
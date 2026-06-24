import cv2

# Select ArUco dictionary
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

# Generate 4 marker images
for marker_id in range(4):
    marker_img = cv2.aruco.generateImageMarker(
        aruco_dict,
        marker_id,
        800
    )

    # Save each marker image
    cv2.imwrite(f"aruco_id_{marker_id}.png", marker_img)
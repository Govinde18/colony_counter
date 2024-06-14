import streamlit as st
import cv2
import numpy as np
from PIL import Image

# Function to detect and crop the plate
def crop_to_plate(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply Gaussian blur
    blur = cv2.GaussianBlur(gray, (11, 11), 0)
    # Detect circles using Hough Circle Transform
    circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, dp=1.2, minDist=100,
                               param1=100, param2=30, minRadius=50, maxRadius=0)
    if circles is not None:
        # Get the largest circle (assuming it's the plate)
        circles = np.round(circles[0, :]).astype("int")
        max_radius = 0
        best_circle = None
        for circle in circles:
            x, y, r = circle
            if r > max_radius:
                max_radius = r
                best_circle = circle
        x, y, r = best_circle
        
        # Expand the bounding box slightly
        padding = 10
        x1 = max(0, x - r - padding)
        y1 = max(0, y - r - padding)
        x2 = min(image.shape[1], x + r + padding)
        y2 = min(image.shape[0], y + r + padding)
        
        # Crop the image to the expanded bounding box
        cropped_image = image[y1:y2, x1:x2]
        return cropped_image
    else:
        return image

# Function to process the image and count colonies
def count_colonies(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply Gaussian blur to reduce noise
    blur = cv2.GaussianBlur(gray, (11, 11), 0)
    
    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    
    # Use morphological operations to remove small noise and separate connected objects
    kernel = np.ones((3, 3), np.uint8)
    closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    sure_bg = cv2.dilate(closing, kernel, iterations=3)
    
    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(sure_bg, cv2.DIST_L2, 5)
    ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
    
    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)
    
    # Marker labelling
    ret, markers = cv2.connectedComponents(sure_fg)
    
    # Add one to all labels so that sure background is not 0, but 1
    markers = markers + 1
    
    # Now, mark the region of unknown with zero
    markers[unknown == 0] = 0
    
    markers = cv2.watershed(image, markers)
    image[markers == -1] = [255, 0, 0]
    
    # Count colonies
    colony_count = ret - 1
    
    # Draw contours for visualization
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    result_image = image.copy()
    cv2.drawContours(result_image, contours, -1, (0, 255, 0), 2)
    
    return colony_count, result_image

# Streamlit app
st.title("Bacterial Colony Counter")

uploaded_file = st.file_uploader("Choose a bacterial plate image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image = np.array(image)
    cropped_image = crop_to_plate(image)
    
    st.image(cropped_image, caption="Cropped Image", use_column_width=True)
    
    if st.button("Count Colonies"):
        colony_count, result_image = count_colonies(cropped_image)
        st.image(result_image, caption=f"Processed Image - {colony_count} colonies found", use_column_width=True)
        st.success(f"Number of colonies: {colony_count}")


import streamlit as st
import cv2
import numpy as np
from PIL import Image

# Function to process the image and count colonies
def count_colonies(image):
    # Convert PIL image to OpenCV format
    image = np.array(image)
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply Gaussian blur
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    # Apply a threshold to get a binary image
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Count colonies (contours)
    colony_count = len(contours)
    
    # Draw contours on the original image for visualization
    result_image = image.copy()
    cv2.drawContours(result_image, contours, -1, (0, 255, 0), 2)
    
    return colony_count, result_image

# Streamlit app
st.title("Bacterial Colony Counter")

uploaded_file = st.file_uploader("Choose a bacterial plate image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    if st.button("Count Colonies"):
        colony_count, result_image = count_colonies(image)
        st.image(result_image, caption=f"Processed Image - {colony_count} colonies found", use_column_width=True)
        st.success(f"Number of colonies: {colony_count}")

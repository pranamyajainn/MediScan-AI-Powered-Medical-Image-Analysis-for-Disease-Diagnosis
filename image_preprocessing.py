import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import graycomatrix, graycoprops
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

image_path = "/Users/pranamyajain/vscode/MediScan-AI-Powered-Medical-Image-Analysis-for-Disease-Diagnosis/mediscan/data/raw/gaussian_filtered_images/gaussian_filtered_images/gaussian_filtered_images/Mild/0a61bddab956.png"

# Load image  OpenCV
image = cv2.imread(image_path)

if image is None:
    print("Error: Image not found. Please check the image path.")
else:
    # Convert the image from BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Display the original image
    plt.imshow(image_rgb)
    plt.title('Original Image')
    plt.axis('off')
    plt.show()

    # Resize the image to 224x224 pixels
    resized_image = cv2.resize(image_rgb, (224, 224))

    # Normalize the image pixel values to the range [0, 1]
    normalized_image = resized_image / 255.0

    # Display the resized and normalized image
    plt.imshow(normalized_image)
    plt.title('Resized and Normalized Image')
    plt.axis('off')
    plt.show()

    # Apply Gaussian blur to reduce noise
    blurred_image = cv2.GaussianBlur(resized_image, (5, 5), 0)

    # Convert the blurred image to grayscale
    gray_image = cv2.cvtColor(blurred_image, cv2.COLOR_RGB2GRAY)

    # Apply binary thresholding
    _, segmented_image = cv2.threshold(gray_image, 128, 255, cv2.THRESH_BINARY)

    # Display the segmented image
    plt.imshow(segmented_image, cmap='gray')
    plt.title('Segmented Image')
    plt.axis('off')
    plt.show()

    # Extract texture features using GLCM
    glcm = graycomatrix(segmented_image, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    dissimilarity = graycoprops(glcm, 'dissimilarity')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    energy = graycoprops(glcm, 'energy')[0, 0]
    correlation = graycoprops(glcm, 'correlation')[0, 0]
    print(f"Texture Features: Contrast: {contrast}, Dissimilarity: {dissimilarity}, Homogeneity: {homogeneity}, Energy: {energy}, Correlation: {correlation}")

    # Extract shape features (contour and area)
    contours, _ = cv2.findContours(segmented_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    area, perimeter = 0, 0  # Default values if no contours found
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)
        perimeter = cv2.arcLength(largest_contour, True)
    print(f"Shape Features: Area: {area}, Perimeter: {perimeter}")

    # Extract color features (color histogram)
    masked_image = cv2.bitwise_and(image_rgb, image_rgb, mask=segmented_image)
    hist_b = cv2.calcHist([masked_image], [0], None, [256], [0, 256])
    hist_g = cv2.calcHist([masked_image], [1], None, [256], [0, 256])
    hist_r = cv2.calcHist([masked_image], [2], None, [256], [0, 256])
    print(f"Color Histogram Features: Blue Channel Histogram (first 5 values): {hist_b.flatten()[:5]}")
    print(f"Color Histogram Features: Green Channel Histogram (first 5 values): {hist_g.flatten()[:5]}")
    print(f"Color Histogram Features: Red Channel Histogram (first 5 values): {hist_r.flatten()[:5]}")

    # Combine features into a feature vector
    features = [
        contrast, dissimilarity, homogeneity, energy, correlation, area, perimeter,
        *hist_b.flatten()[:5], *hist_g.flatten()[:5], *hist_r.flatten()[:5]
    ]

    # Creating a mock dataset by repeating the same sample 100 times
    X = np.tile(features, (100, 1))
    y = np.ones(100)  # Assuming all samples belong to the same class for simplicity

    # Splitting data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a Random Forest Classifier
    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    # Predict and evaluate
    y_pred = model.predict(X_test)
    print(f"Model Accuracy: {accuracy_score(y_test, y_pred)}")

    # Save the trained model
    joblib.dump(model, "trained_model.pkl")
    print("Model saved as 'trained_model.pkl'")

import matplotlib.pyplot as plt
import cv2
import numpy as np

# Assuming 'image_rgb' is your loaded and processed RGB image
# and 'segmented_image' is the mask obtained from your processing

# Mask the original RGB image to focus on the area of interest
masked_image = cv2.bitwise_and(image_rgb, image_rgb, mask=segmented_image)

# Calculate histograms for each color channel
hist_b = cv2.calcHist([masked_image], [0], None, [256], [0, 256])
hist_g = cv2.calcHist([masked_image], [1], None, [256], [0, 256])
hist_r = cv2.calcHist([masked_image], [2], None, [256], [0, 256])

# Plot each histogram in a separate subplot
plt.figure(figsize=(15, 5))  # Adjust the figure size to your needs

# Blue channel
plt.subplot(1, 3, 1)  # 1 row, 3 columns, index 1
plt.plot(hist_b, color='blue')
plt.title('Blue Channel Histogram')
plt.xlabel('Intensity Value')
plt.ylabel('Frequency')
plt.xlim([0, 256])
plt.grid(True)

# Green channel
plt.subplot(1, 3, 2)  # 1 row, 3 columns, index 2
plt.plot(hist_g, color='green')
plt.title('Green Channel Histogram')
plt.xlabel('Intensity Value')
plt.ylabel('Frequency')
plt.xlim([0, 256])
plt.grid(True)

# Red channel
plt.subplot(1, 3, 3)  # 1 row, 3 columns, index 3
plt.plot(hist_r, color='red')
plt.title('Red Channel Histogram')
plt.xlabel('Intensity Value')
plt.ylabel('Frequency')
plt.xlim([0, 256])
plt.grid(True)

plt.tight_layout()  # Adjust subplots to fit into figure area.
plt.show()

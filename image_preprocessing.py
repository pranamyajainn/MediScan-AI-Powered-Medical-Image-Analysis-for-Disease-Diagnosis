import cv2  # Correct OpenCV module
import matplotlib.pyplot as plt

# Load an image using OpenCV
image_path = r"/Users/pranamyajain/vscode/MediScan-AI-Powered-Medical-Image-Analysis-for-Disease-Diagnosis/mediscan/data/raw/gaussian_filtered_images/gaussian_filtered_images/gaussian_filtered_images/Mild/0a61bddab956.png"
image = cv2.imread(image_path)

# Check if the image was loaded successfully
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
    blurred_image = cv2.GaussianBlur(normalized_image, (5, 5), 0)

    # Display the blurred image
    plt.imshow(blurred_image)
    plt.title('Blurred Image')
    plt.axis('off')
    plt.show()

    # Convert the blurred image to grayscale
    gray_image = cv2.cvtColor((blurred_image * 255).astype('uint8'), cv2.COLOR_RGB2GRAY)

    # Apply binary thresholding
    _, segmented_image = cv2.threshold(gray_image, 128, 255, cv2.THRESH_BINARY)

    # Display the segmented image
    plt.imshow(segmented_image, cmap='gray')
    plt.title('Segmented Image')
    plt.axis('off')
    plt.show()

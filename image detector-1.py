import cv2
import numpy as np
import os
import shutil
from tqdm import tqdm  # For progress bar


def preprocess_image(image):
    """
    Preprocess the image by applying a Gaussian blur to reduce noise.
    """
    blurred_image = cv2.GaussianBlur(image, (5, 5), 0)
    return blurred_image


def extract_dominant_colors(image, k=5, top_n=3):
    """
    Extract the top N dominant colors of an image using K-means clustering in LAB color space.
    """
    # Convert the image to LAB color space
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    # Reshape the image data into a 2D array of pixels
    data = lab_image.reshape((-1, 3))
    data = np.float32(data)

    # Apply K-means clustering
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # Get the top N most dominant colors
    unique, counts = np.unique(labels, return_counts=True)
    sorted_indices = np.argsort(-counts)
    dominant_colors = centers[sorted_indices][:top_n]
    return dominant_colors


def calculate_histogram(image):
    """
    Calculate the color histogram of an image in HSV color space.
    """
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv_image], [0, 1], None, [50, 60], [0, 180, 0, 256])
    cv2.normalize(hist, hist)
    return hist


def histogram_similarity(hist1, hist2):
    """
    Compare two histograms using correlation.
    """
    return cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)


def color_distance(color1, color2):
    """
    Calculate the Euclidean distance between two colors.
    """
    return np.linalg.norm(color1 - color2)


def scan_images(input_folder):
    """
    Scan the input folder for valid image files.
    """
    files = [file for file in os.listdir(input_folder) if file.endswith(('.jpg', '.jpeg', '.png'))]
    valid_images = []
    for file in tqdm(files, desc="Scanning Images"):
        filepath = os.path.join(input_folder, file)
        image = cv2.imread(filepath)
        if image is not None:
            valid_images.append(filepath)
        else:
            print(f"Invalid image file skipped: {file}")
    return valid_images


def sort_images_by_color(input_folder, output_folder, k=5, top_n=3, threshold=50.0, hist_threshold=0.7):
    """
    Sort images by their dominant colors, histograms, and save them in grouped folders.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Scan for valid images
    valid_images = scan_images(input_folder)

    if not valid_images:
        print("No valid images found in the input folder.")
        return

    color_groups = []
    histogram_groups = []
    group_folders = []

    # Use tqdm to create a progress bar
    for filepath in tqdm(valid_images, desc="Processing Images"):
        file = os.path.basename(filepath)
        image = cv2.imread(filepath)

        # Preprocess the image
        preprocessed_image = preprocess_image(image)

        # Extract dominant colors
        dominant_colors = extract_dominant_colors(preprocessed_image, k, top_n)

        # Calculate histogram
        hist = calculate_histogram(preprocessed_image)

        # Check if the image matches any existing group
        group_found = False
        for idx, color_group in enumerate(color_groups):
            # Check color similarity
            color_similarities = [color_distance(dominant_color, group_color) for dominant_color, group_color in zip(dominant_colors, color_group)]
            avg_color_similarity = sum(color_similarities) / len(color_similarities)

            # Check histogram similarity
            hist_similarity = histogram_similarity(hist, histogram_groups[idx])

            if avg_color_similarity < threshold and hist_similarity > hist_threshold:
                shutil.move(filepath, os.path.join(group_folders[idx], file))
                group_found = True
                break

        # If no group matches, create a new one
        if not group_found:
            new_group_folder = os.path.join(output_folder, f"group_{len(color_groups)+1}")
            if not os.path.exists(new_group_folder):  # Ensure the folder does not already exist
                os.makedirs(new_group_folder)
            shutil.move(filepath, os.path.join(new_group_folder, file))

            # Save the dominant colors and histogram for the new group
            color_groups.append(dominant_colors)
            histogram_groups.append(hist)
            group_folders.append(new_group_folder)

    print("Images sorted successfully!")


# Input and output folders
input_folder = r"C:\Users\jjy36\Downloads\trial"
output_folder = r"C:\Users\jjy36\Downloads\sorted_images"

# Run the sorting function
sort_images_by_color(input_folder, output_folder, k=10, top_n=5, threshold=30.0, hist_threshold=0.8)

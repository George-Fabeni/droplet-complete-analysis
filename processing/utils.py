# processing/utils.py
import cv2
import numpy as np
import os
import glob
from tkinter import filedialog
import tkinter as tk

def load_image_cv2(filepath):
    """Loads an image using OpenCV, returns BGR NumPy array."""
    img_bgr = cv2.imread(filepath)
    if img_bgr is None:
        raise FileNotFoundError(f"Erro: Não foi possível carregar a imagem: {filepath}")
    return img_bgr

def load_image_from_dialog():
    """Opens a file dialog to select an image and returns its path."""
    root_temp = tk.Tk()
    root_temp.withdraw()
    filepath = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.tiff *.tif *.png")])
    root_temp.destroy()
    return filepath

def get_image_paths_from_folder(folder_path):
    """Gets a sorted list of image file paths from a given folder."""
    if not os.path.isdir(folder_path):
        print(f"A pasta '{folder_path}' não existe.")
        return []
    
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.tiff', '*.tif']
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(glob.glob(os.path.join(folder_path, ext)))
    
    image_paths.sort() # Ensure consistent order
    return image_paths

def rotate_image(image, angle):
    """Rotates an image by a given angle."""
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_img = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    return rotated_img

def create_display_frame(cropped_original_color, prominence_contour, measurements, show_metrics_text=True):
    """Creates a display frame with contour and metrics."""
    display_frame = cropped_original_color.copy()

    if prominence_contour is not None and prominence_contour.shape[0] > 0:
        cv2.drawContours(display_frame, [prominence_contour], -1, (0, 255, 0), 1)
        # Assuming measurements['cX'] and ['cY'] exist from centroid calculation
        if 'cX' in measurements and 'cY' in measurements:
            cv2.circle(display_frame, (int(measurements['cX']), int(measurements['cY'])), 5, (255, 0, 255), -1)
            # Draw centroid line for visual debugging
            cv2.line(display_frame, (int(measurements['cX']), 0), (int(measurements['cX']), cropped_original_color.shape[0]), (0, 255, 255), 1)
        
    if show_metrics_text:
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        font_thickness = 1
        text_color = (255, 255, 255) # White

        # Display all relevant metrics
        metrics_to_display = {
            "Volume": "volume_uL",
            "Base Length": "base_length_mm",
            "Height": "height_mm",
            "Form Factor": "form_factor",
            "Surface Area (Air)": "surface_area_air_mm2",
            "Surface Area (Total)": "surface_area_total_mm2",
            "Angle L": "contact_angle_left_deg",
            "Angle R": "contact_angle_right_deg",
            "Angle Avg": "contact_angle_avg_deg"
        }
        
        y_offset = 20
        for label, key in metrics_to_display.items():
            if key in measurements:
                value = measurements[key]
                if isinstance(value, (int, float)):
                    text = f'{label}: {value:.2f}' if label != "Form Factor" else f'{label}: {value:.2f}'
                    if "Angle" in label: # For angles, often one decimal is enough
                         text = f'{label}: {value:.1f} deg'
                    cv2.putText(display_frame, text, (10, y_offset), font, font_scale, text_color, font_thickness, cv2.LINE_AA)
                    y_offset += 25 # Increment line spacing

    return display_frame
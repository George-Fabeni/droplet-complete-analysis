# processing/utils.py
import cv2
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

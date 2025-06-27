# processing/utils.py
import cv2
import os
import glob
from tkinter import filedialog
import tkinter as tk
import numpy as np

def save_frame_to_video(frames, output_path, fps=30):
    """
    Salva uma lista de frames em um arquivo de vídeo.

    Args:
        frames (list): Uma lista de arrays NumPy, onde cada array é um frame (imagem).
        output_path (str): O caminho completo para o arquivo de vídeo de saída (ex: 'video_final.mp4').
        fps (int, optional): Quadros por segundo (frames per second) do vídeo. Padrão é 30.
    """
    if not frames:
        print("Nenhum frame fornecido para salvar o vídeo.")
        return

    # Obter as dimensões do primeiro frame para inicializar o VideoWriter
    # Assumimos que todos os frames têm as mesmas dimensões
    height, width, layers = frames[0].shape
    
    # Define o codec e cria o objeto VideoWriter
    # 'mp4v' é um bom codec para MP4 que geralmente funciona bem
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    if not out.isOpened():
        print(f"Erro: Não foi possível abrir o arquivo de vídeo para escrita: {output_path}")
        return

    print(f"Iniciando gravação do vídeo com {len(frames)} frames em {output_path}...")
    for i, frame in enumerate(frames):
        # Garante que o frame tem as dimensões corretas (width, height)
        # Se os frames puderem ter tamanhos ligeiramente diferentes, você pode precisar redimensioná-los aqui
        # Ex: frame_resized = cv2.resize(frame, (width, height))
        
        # Certifique-se de que o frame é do tipo correto (uint8)
        if frame.dtype != np.uint8:
            frame = frame.astype(np.uint8)

        out.write(frame)
        if (i + 1) % 100 == 0:
            print(f"  Escrevendo frame {i+1}/{len(frames)}...")

    out.release()
    print(f"Vídeo salvo com sucesso em: {output_path}")
    
    
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

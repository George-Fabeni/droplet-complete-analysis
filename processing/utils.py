# processing/utils.py
import cv2
import os
import glob
from tkinter import filedialog
import tkinter as tk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def save_frame_to_video(frames, output_path, fps=30):
    """
    Saves a list of frames to a video file.

    Args:
        frames (list): NumPy array list, where each array is a frame (image).
        output_path (str): The full path to the output video file
        fps (int, optional): frames per second
    """
    if not frames:
        print("No available frames for saving the video.")
        return

    # Find image shape using the first image
    height, width, layers = frames[0].shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    if not out.isOpened():
        print(f"Error: Couldn't open video file': {output_path}")
        return

    print(f"Starting video generation with {len(frames)} frames in {output_path}...")
    for i, frame in enumerate(frames):

        # Certifies that the frame is in the correct format (uint8)
        if frame.dtype != np.uint8:
            frame = frame.astype(np.uint8)

        out.write(frame)
        if (i + 1) % 100 == 0:
            print(f"  Writting frame {i+1}/{len(frames)}...")

    out.release()
    print(f"Video saved successfully: {output_path}")
    
    
def load_image_cv2(filepath):
    """Loads an image using OpenCV, returns BGR NumPy array."""
    img_bgr = cv2.imread(filepath)
    if img_bgr is None:
        raise FileNotFoundError(f"Erro: Couldn't load the image': {filepath}")
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
        print(f"The folder'{folder_path}' doesnt't exist.")
        return []
    
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.tiff', '*.tif']
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(glob.glob(os.path.join(folder_path, ext)))
    
    image_paths.sort() # Ensure consistent order
    return image_paths


def convert_image_name(image_name):
    # Gets a file name (dd-mm-yy hh-mm-ss.extension)
    img_separeted_extension = image_name.split(".")
    img_without_extension = img_separeted_extension[0]
    parts = img_without_extension.split(' ') # Splits it in two
    date_str_orig = parts[0]  # "dd-mm-yyyy"
    time_str_orig = parts[1]  # "hh_mm_ss"
    
    # 2. Converts to right format
    format_date = date_str_orig.replace('-', '/') # "17/07/2025"
    format_time = time_str_orig.replace('_', ':') # "10:02:29"
    full_string_datetime = format_date + ' ' + format_time

    # Format must be the same as used in the csv measurements file '%d/%m/%Y %H:%M:%S'
    date_time_image = pd.to_datetime(full_string_datetime, format='%d/%m/%Y %H:%M:%S')
    return(date_time_image)


def rotate_image(image, angle):
    """Rotates an image by a given angle."""
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_img = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    return rotated_img

def spreadsheet_reading(base_file_path):

    try:
        df = pd.read_csv(base_file_path, encoding='latin1', sep=';', decimal=',', skiprows=8) # Usar caso seja um csv
        #df = pd.read_excel(base_file_path, [0]) #Usar caso seja um xlmx

        # Nomes das colunas
        col0 = df.columns[0]
        col1 = df.columns[1]
        col2 = df.columns[2]
        col3 = df.columns[3]

        # Limpeza e conversão para numérico para todas as colunas relevantes
        col0_limpa = df[col0].astype(str).str.strip()
        col1_limpa = df[col1].astype(str).str.strip()
        col2_limpa = pd.to_numeric(df[col2].astype(str).str.strip(), errors='coerce')
        col3_limpa = pd.to_numeric(df[col3].astype(str).str.strip(), errors='coerce')

        df_plot = pd.DataFrame({
            'col0': col0_limpa,
            'col1': col1_limpa,
            'col2': col2_limpa,
            'col3': col3_limpa 
        }).dropna()
        
        df_plot['datetime'] = pd.to_datetime(df_plot["col0"] + ' ' + df_plot["col1"], format='%d/%m/%Y %H:%M:%S')

    
        if df_plot.empty:
            print("Não há dados válidos para plotar após a conversão e remoção de valores ausentes.")
            return
        
    except FileNotFoundError:
        print(f"Erro: O arquivo '{base_file_path}' não foi encontrado.")
    except pd.errors.EmptyDataError:
        print(f"Erro: O arquivo '{base_file_path}' está vazio.")
    except pd.errors.ParserError as e:
        print(f"Erro de parsing: Verifique o delimitador (sep=';') e a estrutura das linhas do CSV. Detalhes: {e}")
    except UnicodeDecodeError:
        print(f"Erro de codificação: Não foi possível ler o arquivo '{base_file_path}' com a codificação especificada. Tente 'windows-1252' ou verifique a codificação do arquivo.")
    except KeyError as e:
        print(f"Erro: Coluna não encontrada. Verifique se as colunas estão corretas (índices 0, 1 e 2). Detalhes: {e}")
    except Exception as e:
        print(f"Ocorreu um erro inesperado: {e}")
        
    return df_plot

def plot_graph(df_plot):
        
        xmin, xmax = df_plot["full_time_data"].min(), df_plot["full_time_data"].max()
        ymin1, ymax1 = df_plot['corresponding_mass'].min(), df_plot['corresponding_mass'].max()
        ymin2, ymax2 = df_plot['corresponding_volume'].min(), df_plot['corresponding_volume'].max()
        
        color_y1 = 'tab:red'
        color_y2 = 'tab:blue'
        plt_title = "Graph title"
        
        # Axis y1 config
        fig, ax1 = plt.subplots(figsize=(12, 7)) # Cria a figura e o primeiro eixo (principal)  
        
        ax1.set_xlabel("Date and time", fontsize = 18)
        ax1.set_ylabel("Mass (g)", color=color_y1, fontsize = 18)
        ax1.tick_params(axis='y', labelcolor=color_y1)
        ax1.plot(df_plot["full_time_data"], df_plot["corresponding_mass"], color=color_y1, linewidth=2, label='Axis 1')
        #ax1.plot(measurements["full_time_data"], df_plot["corresponding_volume"], color=color_y1, linewidth=2, label='Axis 1')
        ax1.set_xlim(xmin, xmax) # Definir limite X para ambos os eixos
        ax1.set_ylim(ymin1, ymax1)# Definir limite Y para o eixo principal        
        
        # Axis y2 config
        ax2 = ax1.twinx()
        ax2.set_ylabel("Volume (uL)", color=color_y2, fontsize = 18)
        ax2.tick_params(axis='y', labelcolor=color_y2)
        ax2.plot(df_plot['full_time_data'], df_plot['corresponding_volume'], color=color_y2, linestyle='--', linewidth=2, label='Axis 2')
        ax2.set_ylim(ymin2, ymax2)      
    
        ax1.grid(True, linestyle='--', alpha=0.7) # Adicionar grade

        # Plot graph
        plt.title(plt_title, fontsize=20)  
        plt.tight_layout() 
        plt.show()  

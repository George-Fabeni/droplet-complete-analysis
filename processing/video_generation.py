# processing/video_generation.py
import cv2
import os
import pandas as pd
from datetime import datetime
import numpy as np # Adicionar import do numpy

from processing.utils import load_image_cv2, rotate_image
from processing.image_processing import process_frame_for_droplets # Certifique-se de que esta importação existe
from processing.image_adjustments import apply_adjustments_cv2 # Certifique-se de que esta importação existe
from config.settings import VIDEO_FPS, PX_PER_MM, MM3_PER_UL # Certifique-se de que estas importações existem

def process_images_and_generate_video(
    image_paths, 
    base_image_path, 
    rotation_angle, 
    crop_coords, # [x1, y1, x2, y2]
    output_video_path, 
    output_csv_path,
    brightness, exposure, contrast, highlights, shadows
):
    if not image_paths:
        print("Nenhuma imagem para processar.")
        return

    # Load and process base image once
    base_image_full = load_image_cv2(base_image_path)
    base_image_rotated = rotate_image(base_image_full, rotation_angle)

    # Apply crop to base image
    x1, y1, x2, y2 = crop_coords
    
    # Validate crop_coords against the base_image_rotated dimensions
    h_base, w_base = base_image_rotated.shape[:2]
    # Ensure crop coordinates are within base image bounds
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w_base, x2)
    y2 = min(h_base, y2)

    # Adjust if crop area is invalid after clamping
    if x2 <= x1 or y2 <= y1:
        # Fallback to full image if crop is invalid or zero area
        print(f"Atenção: Coordenadas de corte inválidas após clamping: {crop_coords}. Usando a imagem base completa para o corte.")
        x1, y1, x2, y2 = 0, 0, w_base, h_base
        crop_coords = [x1, y1, x2, y2] # Update crop_coords for subsequent images

    base_image_cropped = base_image_rotated[y1:y2, x1:x2]
    base_image_processed = apply_adjustments_cv2(base_image_cropped, brightness, exposure, contrast, highlights, shadows)


    # Determine video dimensions from the *cropped* base image
    # All frames will be resized to this size for consistency
    height, width = base_image_processed.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Codec for .mp4
    out = cv2.VideoWriter(output_video_path, fourcc, VIDEO_FPS, (width, height))

    measurements_data = []

    print(f"Iniciando processamento de {len(image_paths)} imagens e geração de vídeo...")

    for i, img_path in enumerate(image_paths):
        try:
            print(f"Processando imagem {i+1}/{len(image_paths)}: {os.path.basename(img_path)}")

            current_image_full = load_image_cv2(img_path)
            current_image_rotated = rotate_image(current_image_full, rotation_angle)

            # --- APLICAR CORTE DE FORMA ROBUSTA AQUI ---
            # Use as coordenadas de corte que foram (possivelmente) ajustadas para a imagem base
            x1, y1, x2, y2 = crop_coords 

            # Validar e ajustar as coordenadas de corte para a imagem atual
            h_current, w_current = current_image_rotated.shape[:2]
            
            # Clamp crop_coords to current image dimensions
            safe_x1 = max(0, x1)
            safe_y1 = max(0, y1)
            safe_x2 = min(w_current, x2)
            safe_y2 = min(h_current, y2)

            # If the calculated safe crop area is invalid, use the full current image
            if safe_x2 <= safe_x1 or safe_y2 <= safe_y1:
                print(f"Atenção: Área de corte inválida para {os.path.basename(img_path)}. Usando a imagem completa.")
                current_image_cropped = current_image_rotated.copy()
            else:
                current_image_cropped = current_image_rotated[safe_y1:safe_y2, safe_x1:safe_x2].copy()
            # --- FIM DO CORTE ROBUSTO ---


            # Aplicar ajustes de imagem
            current_image_adjusted = apply_adjustments_cv2(
                current_image_cropped, brightness, exposure, contrast, highlights, shadows
            )

            # Redimensionar para o tamanho do vídeo (definido pela imagem base processada)
            # Isso é crucial para que todos os frames tenham o mesmo tamanho
            current_image_final = cv2.resize(current_image_adjusted, (width, height))

            # Processar o frame
            result_frame, frame_data = process_frame_for_droplets(
                current_image_final, base_image_processed, i + 1, PX_PER_MM, MM3_PER_UL
            )
            
            out.write(result_frame)
            if frame_data: # Ensure frame_data is not empty
                measurements_data.append(frame_data)

        except Exception as e:
            print(f"Erro ao processar {os.path.basename(img_path)}: {e}")
            continue # Continue to next image even if one fails

    out.release()

    if measurements_data:
        df = pd.DataFrame(measurements_data)
        df.to_csv(output_csv_path, index=False)
        print(f"Dados de medição salvos em: {output_csv_path}")
    else:
        print("Nenhum dado de medição foi coletado.")

    print(f"Vídeo salvo em: {output_video_path}")
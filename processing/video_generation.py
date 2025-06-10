# processing/video_generation.py
import cv2
import os
import pandas as pd
from datetime import datetime # Certifique-se de que esta linha está presente
import numpy as np

from processing.utils import load_image_cv2, rotate_image
# IMPORTAR SUAS FUNÇÕES ESPECÍFICAS AQUI
from processing.image_processing import segment_drop, calculate_measurements # <--- ATUALIZADO AQUI
from processing.image_adjustments import apply_adjustments_cv2
from config.settings import VIDEO_FPS, PX_PER_MM, MM3_PER_UL, THRESHOLD_VALUE_DIFFERENCE, KERNEL_BLUR_SIZE, KERNEL_MORPH_SIZE # <--- Adicionar parâmetros de processamento

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

    # Load and process base image once (full resolution, rotated)
    base_image_full_res = load_image_cv2(base_image_path)
    base_image_rotated = rotate_image(base_image_full_res, rotation_angle)

    # Base image is used for difference, but the video dimensions will come from the *first* cropped image.
    # We will pass the crop_coords directly to segment_drop for each image.

    measurements_data = [] # Lista para armazenar os dados de medição

    # Determine video dimensions from the *first* image's crop, or fallback to default
    # This loop is just to get the first valid frame's dimensions for video writer initialization
    first_frame_dims = None
    for i, img_path in enumerate(image_paths):
        current_image_full_res = load_image_cv2(img_path)
        current_image_rotated = rotate_image(current_image_full_res, rotation_angle)
        
        # Call segment_drop to get the cropped_current_color (the actual image content after crop)
        # We don't need mask_full or prominence_contour here, just the dimensions of the cropped image.
        _, _, temp_cropped_color = segment_drop(
            current_image_rotated, base_image_rotated, crop_coords,
            THRESHOLD_VALUE_DIFFERENCE, KERNEL_BLUR_SIZE, KERNEL_MORPH_SIZE, debug_plots=False # No debug plots for initial dim
        )
        
        if temp_cropped_color is not None and temp_cropped_color.shape[0] > 0 and temp_cropped_color.shape[1] > 0:
            first_frame_dims = temp_cropped_color.shape[:2] # height, width
            break # Got dimensions, exit loop
        elif i == len(image_paths) - 1:
            print("Nenhuma imagem válida para determinar as dimensões do vídeo. Usando fallback.")
            first_frame_dims = (720, 1280) # Fallback if no valid image/crop found

    height, width = first_frame_dims
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Codec for .mp4
    out = cv2.VideoWriter(output_video_path, fourcc, VIDEO_FPS, (width, height))

    print(f"Iniciando processamento de {len(image_paths)} imagens e geração de vídeo...")

    for i, img_path in enumerate(image_paths):
        try:
            print(f"Processando imagem {i+1}/{len(image_paths)}")

            current_image_full_res = load_image_cv2(img_path)
            current_image_rotated = rotate_image(current_image_full_res, rotation_angle)

            # --- SEGMENTAR A GOTA USANDO SUA LÓGICA ---
            mask_full, prominence_contour, cropped_current_color = segment_drop(
                current_image_rotated, base_image_rotated, crop_coords,
                THRESHOLD_VALUE_DIFFERENCE, KERNEL_BLUR_SIZE, KERNEL_MORPH_SIZE, debug_plots=False
            )

            if cropped_current_color is None or cropped_current_color.shape[0] == 0 or cropped_current_color.shape[1] == 0:
                print(f"Atenção: Imagem {os.path.basename(img_path)} resultou em um frame cropped vazio. Pulando processamento.")
                continue # Skip this frame if it's empty after segment_drop

            # Aplicar ajustes de imagem (Brilho, Contraste, etc.)
            # Estes ajustes são aplicados no frame *já cortado*
            current_image_adjusted = apply_adjustments_cv2(
                cropped_current_color, brightness, exposure, contrast, highlights, shadows
            )

            # Redimensionar para o tamanho do vídeo (definido no início)
            # Isso é crucial para que todos os frames tenham o mesmo tamanho
            processed_frame_for_video = cv2.resize(current_image_adjusted, (width, height))


            # --- CALCULAR MEDIÇÕES USANDO SUA LÓGICA ---
            measurements = calculate_measurements(mask_full, prominence_contour, PX_PER_MM, MM3_PER_UL)
            
            # Adicionar dados específicos do frame ao dicionário de medições
            measurements['frame_number'] = i + 1
            measurements['image_name'] = os.path.basename(img_path)

            measurements_data.append(measurements)

            # --- DESENHAR INFORMAÇÕES NO VÍDEO (Se desejar) ---
            # Você pode querer desenhar o contorno, o centroide, e as medições no `processed_frame_for_video`
            # antes de escrevê-lo no vídeo.
            if prominence_contour is not None and prominence_contour.shape[0] > 0:

                scale_w = width / cropped_current_color.shape[1]
                scale_h = height / cropped_current_color.shape[0]
                
                scaled_contour = prominence_contour.copy()
                scaled_contour[:, 0, 0] = (scaled_contour[:, 0, 0] * scale_w).astype(int)
                scaled_contour[:, 0, 1] = (scaled_contour[:, 0, 1] * scale_h).astype(int)
                
                cv2.drawContours(processed_frame_for_video, [scaled_contour], -1, (0, 255, 0), 2) # Verde
                
                # Desenhar centroide
                # Se o centroide (cX, cY) foi calculado em relação ao cropped_current_color, 
                # precisamos escalá-lo também
                scaled_cX = int(measurements['cX'] * scale_w)
                scaled_cY = int(measurements['cY'] * scale_h)
                cv2.circle(processed_frame_for_video, (scaled_cX, scaled_cY), 5, (0, 0, 255), -1) # Vermelho
                
                # Desenhar texto com medições
                
                '''info_text = (f"Frame: {i+1} | Area: {measurements['area_pixels']:.0f}px | "
                             f"Vol: {measurements['volume_uL']:.3f}uL | "
                             f"Diam_B: {measurements['base_length_mm']:.2f}mm | "
                             f"Height: {measurements['height_mm']:.2f}mm")
                cv2.putText(processed_frame_for_video, info_text, (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA) '''
                
                # ------------- TESTE ------------------ 
                # DESCOMENTAR PARTE ANTERIOR SE O TESTE DER ERRADO
                
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.6
                font_thickness = 1
                text_color = (0, 0, 0)

                cv2.putText(processed_frame_for_video, f'Area: {measurements["surface_area_total_mm2"]:.2f} mm^2', (10, 20),
                            font, font_scale, text_color, font_thickness, cv2.LINE_AA)
                cv2.putText(processed_frame_for_video, f'Volume: {measurements["volume_uL"]:.2f} uL', (10, 40),
                            font, font_scale, text_color, font_thickness, cv2.LINE_AA)
                cv2.putText(processed_frame_for_video, f'Base Length: {measurements["base_length_mm"]:.2f} mm', (10, 60),
                            font, font_scale, text_color, font_thickness, cv2.LINE_AA)
                cv2.putText(processed_frame_for_video, f'Height: {measurements["height_mm"]:.2f} mm', (10, 80),
                            font, font_scale, text_color, font_thickness, cv2.LINE_AA)
                cv2.putText(processed_frame_for_video, f'Form factor: {measurements["form_factor"]:.2f}', (10, 100),
                            font, font_scale, text_color, font_thickness, cv2.LINE_AA)
                
                #---------- FIM DO TESTE ---------------


            out.write(processed_frame_for_video)

        except Exception as e:
            print(f"Erro ao processar {os.path.basename(img_path)}: {e}")
            # Em caso de erro, adicione um frame vazio para não quebrar o vídeo
            if first_frame_dims:
                blank_frame = np.zeros((height, width, 3), dtype=np.uint8)
                out.write(blank_frame)
            continue # Continue para a próxima imagem

    out.release()

    # --- REMOVER GERAÇÃO DE CSV SE NÃO FOR ÚTIL ---
    if measurements_data:
        df = pd.DataFrame(measurements_data)
        df.to_csv(output_csv_path, index=False)
        print(f"Dados de medição salvos em: {output_csv_path}")
    else:
        print("Nenhum dado de medição foi coletado.")
    # --- FIM DA REMOÇÃO ---

    print(f"Vídeo salvo em: {output_video_path}")
import cv2
import numpy as np
import os

from config.settings import THRESHOLD_VALUE_DIFFERENCE, KERNEL_BLUR_SIZE, KERNEL_MORPH_SIZE, VIDEO_FPS, PX_PER_MM, MM3_PER_UL, INITIAL_CONC
from processing.utils import load_image_cv2, rotate_image, save_frame_to_video # Certifique-se de que save_frame_to_video existe
from processing.image_adjustments import apply_adjustments_cv2


def calculate_image_difference(base_image_cv2, current_image_cv2, crop_coords, debug_plots=False):
    """
    Calcula a diferença entre duas imagens (já cortadas) e retorna a imagem de diferença
    e a imagem atual colorida (também já cortada).
    Args:
        base_image_cv2 (np.array): Imagem de base OpenCV (BGR, já cortada).
        current_image_cv2 (np.array): Imagem atual OpenCV (BGR, já cortada).
        crop_coords (list): Coordenadas de corte [x1, y1, x2, y2]. Não é mais usado para cortar,
                             mas mantido para compatibilidade de assinatura se necessário.
        debug_plots (bool): Se True, exibe janelas de depuração.
    Returns:
        tuple: (diferenca_raw, cropped_original_color)
    """
    # As imagens já devem vir cortadas de `_load_and_display_current_image` na GUI.
    # No processamento em lote, o corte será feito em `process_images_and_generate_video`.
    
    # Conversão para escala de cinza das imagens já cortadas
    gray_base = cv2.cvtColor(base_image_cv2, cv2.COLOR_BGR2GRAY)
    gray_current = cv2.cvtColor(current_image_cv2, cv2.COLOR_BGR2GRAY)

    # ------------ CORREÇÃO DE PROBLEMA COM REFLEXOS NA GOTA -------------------
    # Os reflexos nas gotas causavam problemas na identificação dos contornos.
    # O código abaixo corrige isso ao aplicar um thresholding inicial,
    # com espessura "thickness=cv2.FILLED", que preenche tudo
    _, thresholded_base = cv2.threshold(gray_base, THRESHOLD_VALUE_DIFFERENCE, 255, cv2.THRESH_BINARY_INV)
    _, thresholded_current = cv2.threshold(gray_current, THRESHOLD_VALUE_DIFFERENCE, 255, cv2.THRESH_BINARY_INV)

    # Encontrar o maior contorno e preencher
    base_filled = np.ones_like(gray_base) * 255 # fundo branco
    contours_base, _ = cv2.findContours(thresholded_base, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours_base:
        largest_contour_base = max(contours_base, key=cv2.contourArea)
        cv2.drawContours(base_filled, [largest_contour_base], -1, (0,), thickness=cv2.FILLED)

    current_filled = np.ones_like(gray_current) * 255 # fundo branco
    contours_current, _ = cv2.findContours(thresholded_current, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours_current:
        largest_contour_current = max(contours_current, key=cv2.contourArea)
        cv2.drawContours(current_filled, [largest_contour_current], -1, (0,), thickness=cv2.FILLED)

    diferenca_raw = cv2.absdiff(current_filled, base_filled)

    # ------------ FIM DA CORREÇÃO -------------------

    if debug_plots:
        cv2.imshow("Difference Raw (Cropped)", diferenca_raw)
        # cv2.waitKey(1) # Small wait to allow window to render

    # Retorna a imagem de diferença e a imagem atual colorida (já cortada)
    return diferenca_raw, current_image_cv2 # current_image_cv2 já é a cropped_original_color neste contexto


def process_difference_image(diferenca_raw, threshold_value_difference, kernel_blur_size, kernel_morph_size, debug_plots=False):
    """
    Processa a imagem de diferença para obter uma máscara binária e o contorno da proeminência.
    Args:
        diferenca_raw (np.array): Imagem de diferença em escala de cinza.
        threshold_value_difference (int): Valor de limiar para binarização.
        kernel_blur_size (tuple): Tamanho do kernel para o desfoque Gaussiano.
        kernel_morph_size (int): Tamanho do kernel para operações morfológicas.
        debug_plots (bool): Se True, exibe janelas de depuração.
    Returns:
        tuple: (mask_full, prominence_contour)
    """
    # Desfoque Gaussiano
    blurred_diff = cv2.GaussianBlur(diferenca_raw, kernel_blur_size, 0)

    # Binarização
    _, thresh_diff = cv2.threshold(blurred_diff, threshold_value_difference, 255, cv2.THRESH_BINARY)

    kernel = np.ones((kernel_morph_size, kernel_morph_size), np.uint8)
    mask_open = cv2.morphologyEx(thresh_diff, cv2.MORPH_OPEN, kernel)
    mask_full = cv2.morphologyEx(mask_open, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask_full, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    prominence_contour = None
    if contours:
        # Encontrar o maior contorno por área
        prominence_contour = max(contours, key=cv2.contourArea)

    if debug_plots:
        cv2.imshow("Processed Mask", mask_full)

    return mask_full, prominence_contour


def segment_drop(current_image_cropped, base_image_cropped, crop_coords,
                 threshold_value_difference=25, kernel_blur_size=(5, 5), kernel_morph_size=5, debug_plots=False):
    """
    Segmenta a gota na imagem atual usando a imagem de base para diferença.
    Args:
        current_image_cropped (np.array): Imagem atual (já cortada) OpenCV (BGR).
        base_image_cropped (np.array): Imagem de base (já cortada) OpenCV (BGR).
        crop_coords (list): Coordenadas de corte [x1, y1, x2, y2]. Não é mais usado para cortar.
        threshold_value_difference (int): Limiar para a binarização da imagem de diferença.
        kernel_blur_size (tuple): Tamanho do kernel para desfoque.
        kernel_morph_size (int): Tamanho do kernel para morfologia.
        debug_plots (bool): Se True, exibe janelas de depuração.
    Returns:
        tuple: (mask_full, prominence_contour, processed_image_display)
    """
    if current_image_cropped is None or base_image_cropped is None:
        # Retornar máscaras e imagens em branco se alguma imagem for None
        dummy_mask = np.zeros((100, 100), dtype=np.uint8) # Tamanho dummy
        dummy_color = np.zeros((100, 100, 3), dtype=np.uint8)
        return dummy_mask, None, dummy_color

    # Chama calculate_image_difference com as imagens já cortadas
    diferenca_raw, processed_image_display = calculate_image_difference(
        base_image_cropped, current_image_cropped, crop_coords, debug_plots
    )

    # Processa a imagem de diferença para obter a máscara e o contorno
    mask_full, prominence_contour = process_difference_image(
        diferenca_raw, threshold_value_difference, kernel_blur_size, kernel_morph_size, debug_plots
    )

    return mask_full, prominence_contour, processed_image_display


def calculate_measurements(mask_prominence_full, prominence_contour, px_per_mm, mm3_per_ul, initial_volume=None):
    """
    Calcula as medições da gota a partir da máscara e contorno.

    Args:
        mask_prominence_full (np.array): A máscara binária da gota.
        prominence_contour (np.array): O contorno da gota.
        px_per_mm (float): Pixels por milímetro.
        mm3_per_ul (float): Fator de conversão de mm³ para µL.

    Returns:
        dict: Dicionário com todas as medições calculadas.
    """
    measurements = {
        'area_pixels': 0, 'cX': 0, 'cY': 0, 'volume_pixels3': 0, 'volume_uL': 0,
        'base_length_pixels': 0, 'base_length_mm': 0,
        'height_pixels': 0, 'height_mm': 0, 
        'form_factor': 0, 
        'surface_area_total_pixels2': 0, 'surface_area_total_mm2': 0,
        'base_area_pixels2': 0, 'base_area_mm2': 0,
        'surface_area_air_pixels2': 0, 'surface_area_air_mm2': 0, "concentration": 0
    }

    if prominence_contour is None or prominence_contour.shape[0] == 0 or np.sum(mask_prominence_full) == 0:
        return measurements

    measurements['area_pixels'] = np.sum(mask_prominence_full == 255)

    M_full = cv2.moments(mask_prominence_full)
    if M_full["m00"] != 0:
        measurements['cX'] = int(M_full["m10"] / M_full["m00"])
        measurements['cY'] = int(M_full["m01"] / M_full["m00"])

    # Calculate volume using Pappus-Guldinus theorem (Volume = 2 * pi * x_bar * Area)
    if M_full["m00"] != 0:
        area_2d_pixels = M_full["m00"] / 255.0 # Convert sum of pixel values (0 or 255) to actual pixel count
        
        # Create a mask for the right half of the droplet
        mask_right_half = np.zeros_like(mask_prominence_full)
        # Ensure cX is within bounds
        # Use measurements['cX'] which is already an int.
        # Clamp to avoid index out of bounds in case of very small/thin mask
        clamped_cX = np.clip(measurements['cX'], 0, mask_prominence_full.shape[1] -1) 
        mask_right_half[:, clamped_cX:] = mask_prominence_full[:, clamped_cX:]

        M_right_half = cv2.moments(mask_right_half)
        if M_right_half["m00"] != 0: # If there's area in the right half
            area_right_half_pixels = M_right_half["m00"] / 255.0
            cX_right_half = (M_right_half["m10"] / M_right_half["m00"])
            x_bar_right_half = abs(cX_right_half - clamped_cX) # Distance from centroid of half to the axis
            
            measurements['volume_pixels3'] = 2 * np.pi * x_bar_right_half * area_right_half_pixels
        else:
            measurements['volume_pixels3'] = 0 # No right half, no volume calculation possible this way

    if measurements['volume_pixels3'] > 0 and px_per_mm > 0:
        volume_mm3 = measurements['volume_pixels3'] / (px_per_mm ** 3)
        measurements['volume_uL'] = volume_mm3 * mm3_per_ul

    # Calculate Base Length (Max X - Min X of the entire contour)
    # Ensure prominence_contour is not None and has points
    if prominence_contour is not None and prominence_contour.shape[0] > 0:
        x_coords = prominence_contour[:, 0, 0]
        measurements['base_length_pixels'] = np.max(x_coords) - np.min(x_coords)
        measurements['base_length_mm'] = measurements['base_length_pixels'] / PX_PER_MM

        # Calculate Height (Max Y - Min Y of the entire contour)
        y_coords = prominence_contour[:, 0, 1]
        measurements['height_pixels'] = np.max(y_coords) - np.min(y_coords)
        measurements['height_mm'] = measurements['height_pixels'] / PX_PER_MM

        # Calculate Form Factor
        if measurements['base_length_mm'] > 0:
            measurements['form_factor'] = measurements['height_mm'] / measurements['base_length_mm']
        else:
            measurements['form_factor'] = 0

        # Calculate Surface Area
        total_perimeter_revolved_pixels2 = 0
        for i in range(prominence_contour.shape[0]):
            p1 = prominence_contour[i, 0]
            p2 = prominence_contour[(i + 1) % prominence_contour.shape[0], 0]

            segment_length = np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
            mid_x = (p1[0] + p2[0]) / 2
            radius = abs(mid_x - measurements['cX']) # Distance from midpoint to centroid X
            
            total_perimeter_revolved_pixels2 += 2 * np.pi * radius * segment_length

        measurements['surface_area_total_pixels2'] = total_perimeter_revolved_pixels2 / 2.0
        measurements['surface_area_total_mm2'] = measurements['surface_area_total_pixels2'] / (PX_PER_MM ** 2)

        # Calculate Base Area (assuming circular base based on base_length_pixels as diameter)
        if measurements['base_length_pixels'] > 0:
            base_radius_pixels = measurements['base_length_pixels'] / 2.0
            measurements['base_area_pixels2'] = np.pi * (base_radius_pixels ** 2)
            measurements['base_area_mm2'] = measurements['base_area_pixels2'] / (PX_PER_MM ** 2)
        
        # Calculate Surface Area in contact with air (Total - Base)
        measurements['surface_area_air_pixels2'] = measurements['surface_area_total_pixels2'] - measurements['base_area_pixels2']
        measurements['surface_area_air_mm2'] = measurements['surface_area_total_mm2'] - measurements['base_area_mm2']
        
        
        # Calculate concentration
        
        if initial_volume and initial_volume != 0:
            measurements["concentration"] = INITIAL_CONC * ( initial_volume / measurements["volume_uL"])
        else:
            measurements["concentration"] = INITIAL_CONC 
            

    return measurements


def process_images_and_generate_video(image_paths, base_image_full_res_uncropped, rotation_angle, crop_coords,
                                      output_video_path, output_csv_path,
                                      brightness, exposure, contrast, highlights, shadows):
    """
    Processa uma série de imagens para análise de gotas e gera um vídeo de saída.
    Args:
        image_paths (list): Lista de caminhos para as imagens a serem processadas.
        base_image_full_res_uncropped (np.array): Imagem de base full-res, já rotacionada.
        rotation_angle (float): Ângulo de rotação aplicado.
        crop_coords (list): Coordenadas de corte [x1, y1, x2, y2].
        output_video_path (str): Caminho para salvar o vídeo de saída.
        output_csv_path (str): Caminho para salvar o arquivo CSV de medições.
        brightness (float): Ajuste de brilho.
        exposure (float): Ajuste de exposição.
        contrast (float): Ajuste de contraste.
        highlights (float): Ajuste de realces.
        shadows (float): Ajuste de sombras.
    """
    all_measurements = []
    video_frames = []

    # Obter as coordenadas de corte seguras
    x1, y1, x2, y2 = crop_coords
    
    # Pre-corta e ajusta a imagem de base uma vez
    base_image_cropped_for_processing = None
    if base_image_full_res_uncropped is not None:
        h_base, w_base = base_image_full_res_uncropped.shape[:2]
        safe_x1 = max(0, min(x1, w_base))
        safe_y1 = max(0, min(y1, h_base))
        safe_x2 = max(0, min(x2, w_base))
        safe_y2 = max(0, min(y2, h_base))

        if safe_x2 > safe_x1 and safe_y2 > safe_y1:
            base_image_cropped_raw = base_image_full_res_uncropped[safe_y1:safe_y2, safe_x1:safe_x2].copy()
        else:
            base_image_cropped_raw = base_image_full_res_uncropped.copy()
        
        base_image_cropped_for_processing = apply_adjustments_cv2(
            base_image_cropped_raw, brightness, exposure, contrast, highlights, shadows
        )
        

    for i, img_path in enumerate(image_paths):
        print(f"Processing image {i+1}/{len(image_paths)}: {os.path.basename(img_path)}")

        try:
            current_image_full_res = load_image_cv2(img_path)
            current_image_full_res_rotated = rotate_image(current_image_full_res, rotation_angle)

            # Aplica o mesmo corte à imagem atual
            h_curr, w_curr = current_image_full_res_rotated.shape[:2]
            safe_x1_curr = max(0, min(x1, w_curr))
            safe_y1_curr = max(0, min(y1, h_curr))
            safe_x2_curr = max(0, min(x2, w_curr))
            safe_y2_curr = min(h_curr, y2) # Use h_curr, w_curr directly for clipping max

            if safe_x2_curr > safe_x1_curr and safe_y2_curr > safe_y1_curr:
                current_image_cropped_raw = current_image_full_res_rotated[safe_y1_curr:safe_y2_curr, safe_x1_curr:safe_x2_curr].copy()
            else:
                current_image_cropped_raw = current_image_full_res_rotated.copy()

            # Aplica os ajustes à imagem atual JÁ CORTADA
            current_image_cropped_adjusted = apply_adjustments_cv2(
                current_image_cropped_raw, brightness, exposure, contrast, highlights, shadows
            )

            # Agora, `segment_drop` recebe as imagens já cortadas e ajustadas
            mask, contour, processed_display_image = segment_drop(
                current_image_cropped_adjusted,
                base_image_cropped_for_processing, # Usa a base já ajustada
                crop_coords, # crop_coords é ignorado dentro de segment_drop para corte, mas mantido na assinatura
                THRESHOLD_VALUE_DIFFERENCE, KERNEL_BLUR_SIZE, KERNEL_MORPH_SIZE
            )

            if i == 0:
                measurements = calculate_measurements(mask, contour, PX_PER_MM, MM3_PER_UL)
                initial_volume = measurements["volume_uL"]
            else:
                measurements = calculate_measurements(mask, contour, PX_PER_MM, MM3_PER_UL, initial_volume)

            all_measurements.append(measurements)
            
            # Prepare frame for video: Draw contour on processed_display_image
            if processed_display_image is not None and contour is not None and len(contour) > 0:
                cv2.drawContours(processed_display_image, [contour], -1, (0, 255, 0), 2) # Green contour
                
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            font_thickness = 1
            text_color = (0, 0, 0)

            cv2.putText(processed_display_image, f'Area: {measurements["surface_area_air_mm2"]:.2f} mm^2', (5, 20),
                        font, font_scale, text_color, font_thickness, cv2.LINE_AA)
            cv2.putText(processed_display_image, f'Volume: {measurements["volume_uL"]:.2f} uL', (5, 40),
                        font, font_scale, text_color, font_thickness, cv2.LINE_AA)
            cv2.putText(processed_display_image, f'Base Length: {measurements["base_length_mm"]:.2f} mm', (5, 60),
                        font, font_scale, text_color, font_thickness, cv2.LINE_AA)
            cv2.putText(processed_display_image, f'Height: {measurements["height_mm"]:.2f} mm', (5, 80),
                        font, font_scale, text_color, font_thickness, cv2.LINE_AA)
            cv2.putText(processed_display_image, f'Form factor: {measurements["form_factor"]:.2f}', (5, 100),
                        font, font_scale, text_color, font_thickness, cv2.LINE_AA)
            cv2.putText(processed_display_image, f'Concentration: {measurements["concentration"]:.2f}%', (5, 120),
                        font, font_scale, text_color, font_thickness, cv2.LINE_AA)

            # Redimensionar para o vídeo se necessário (se não for a resolução desejada)
            # ou simplesmente garantir que as dimensões sejam consistentes
            if processed_display_image is not None:
                video_frames.append(processed_display_image)
            else:
                # Add a blank frame if processing failed for some reason
                blank_frame_dims = (crop_coords[3] - crop_coords[1], crop_coords[2] - crop_coords[0], 3)
                if blank_frame_dims[0] <=0 or blank_frame_dims[1] <=0: # Fallback for invalid crop
                     blank_frame_dims = (600, 800, 3) # Default size if crop is invalid
                video_frames.append(np.zeros(blank_frame_dims, dtype=np.uint8))


        except Exception as e:
            print(f"Error processing {os.path.basename(img_path)}: {e}")
            # Add a blank frame or original image if processing fails
            blank_frame_dims = (crop_coords[3] - crop_coords[1], crop_coords[2] - crop_coords[0], 3)
            if blank_frame_dims[0] <=0 or blank_frame_dims[1] <=0: # Fallback for invalid crop
                blank_frame_dims = (600, 800, 3) # Default size if crop is invalid
            video_frames.append(np.zeros(blank_frame_dims, dtype=np.uint8))
            all_measurements.append({
                'image_name': os.path.basename(img_path),
                'area_mm2': 0.0, 'volume_ul': 0.0, 'perimeter_mm': 0.0, 'circularity': 0.0
            })

    # Save measurements to CSV
    if all_measurements:
        import pandas as pd # Ensure pandas is imported
        df = pd.DataFrame(all_measurements)
        df.to_csv(output_csv_path, index=False)
        print(f"Measurements saved to {output_csv_path}")

    # Save video
    if video_frames:
        save_frame_to_video(video_frames, output_video_path, fps=VIDEO_FPS)
        print(f"Video saved to {output_video_path}")
    else:
        print("No frames to save for the video.")
# processing/image_processing.py
import cv2
import numpy as np
from config.settings import PX_PER_MM, INITIAL_CONC
from config.settings import THRESHOLD_VALUE_DIFFERENCE


def calculate_image_difference(base_image_cv2, current_image_cv2, crop_coords, debug_plots=False):
    """
    Calcula a diferença entre a imagem atual e a imagem base, aplicando o corte.

    Args:
        base_image_cv2 (np.array): A imagem base (já rotacionada).
        current_image_cv2 (np.array): A imagem atual (já rotacionada).
        crop_coords (list): [x1, y1, x2, y2] coordenadas de corte.
        debug_plots (bool): Se True, mostra plots para depuração.

    Returns:
        tuple: (diferenca_raw, cropped_original_color)
               Retorna (None, None) se o corte for inválido para as imagens.
    """
    x1, y1, x2, y2 = crop_coords

    # Valide e clamp as coordenadas de corte para ambas as imagens
    # Assumimos que base_image_cv2 e current_image_cv2 já foram rotacionadas.
    # Precisamos pegar as dimensões delas INDIVIDUALMENTE para garantir que o corte seja seguro.

    h_base, w_base = base_image_cv2.shape[:2]
    h_current, w_current = current_image_cv2.shape[:2]

    # Use as coordenadas de corte calculadas pelo usuário.
    # No entanto, clamp-as aos limites da imagem atual.
    # O crop_coords já foi validado e ajustado no main_window para a imagem que foi exibida.
    # Aqui, garantimos que ele não estoure os limites da imagem que está sendo processada.
    safe_x1 = max(0, x1)
    safe_y1 = max(0, y1)
    safe_x2 = min(w_current, x2) # Min com a largura da imagem atual
    safe_y2 = min(h_current, y2) # Min com a altura da imagem atual

    # Se a área de corte resultante for inválida após o clamping, retorne None
    if safe_x2 <= safe_x1 or safe_y2 <= safe_y1:
        print(f"Warning: Calculated crop area [{safe_x1},{safe_y1},{safe_x2},{safe_y2}] is invalid for current image dimensions ({w_current}x{h_current}). Returning dummy mask.")
        return None, None

    # Aplica o mesmo corte tanto para a imagem base quanto para a imagem atual
    cropped_base = base_image_cv2[safe_y1:safe_y2, safe_x1:safe_x2]
    cropped_current = current_image_cv2[safe_y1:safe_y2, safe_x1:safe_x2]

    cropped_original_color = current_image_cv2[safe_y1:safe_y2, safe_x1:safe_x2].copy() # Cópia para o output

    # Converte para escala de cinza para calcular a diferença
    gray_base = cv2.cvtColor(cropped_base, cv2.COLOR_BGR2GRAY)
    gray_current = cv2.cvtColor(cropped_current, cv2.COLOR_BGR2GRAY)
    
    # ------------ CORREÇÃO DE PROBLEMA COM REFLEXOS NA GOTA -------------------
    # Os reflexos nas gotas causavam problemas na identificação dos contornos. 
    # O código abaixo corrige isso ao aplicar um thresholding inicial, 
    # com espessura "thickness=cv2.FILLED, que preenche tudo que está
    # abaixo com pixels pretos, eliminando os reflexos.
    _, thresholded_base = cv2.threshold(gray_base, THRESHOLD_VALUE_DIFFERENCE, 255, cv2.THRESH_BINARY_INV)
    _, thresholded_current = cv2.threshold(gray_current, THRESHOLD_VALUE_DIFFERENCE, 255, cv2.THRESH_BINARY_INV)
    contours_base, _ = cv2.findContours(thresholded_base, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_current, _ = cv2.findContours(thresholded_current, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    base_filled = np.ones_like(gray_base) * 255  # fundo branco
    if contours_base:
        largest_contour = max(contours_base, key=cv2.contourArea)
        # Desenhar a gota em preto sobre fundo branco
        cv2.drawContours(base_filled, [largest_contour], -1, (0,), thickness=cv2.FILLED)

    current_filled = np.ones_like(gray_current) * 255  # fundo branco
    if contours_current:
        largest_contour = max(contours_current, key=cv2.contourArea)
        # Desenhar a gota em preto sobre fundo branco
        cv2.drawContours(current_filled, [largest_contour], -1, (0,), thickness=cv2.FILLED)
        
        
    diferenca_raw = cv2.absdiff(current_filled, base_filled)
        
    # ------------ FIM DA CORREÇÃO  -------------------

    if debug_plots:
        cv2.imshow("Difference Raw (Cropped)", diferenca_raw)
        # cv2.waitKey(1) # Small wait to allow window to render

    return diferenca_raw, cropped_original_color


def process_difference_image(diferenca_raw, threshold_value_difference, kernel_blur_size, kernel_morph_size, debug_plots=False):
    """
    Processa a imagem de diferença para obter a máscara da gota e seu contorno de proeminência.

    Args:
        diferenca_raw (np.array): Imagem de diferença em escala de cinza.
        threshold_value_difference (int): Valor de limiar para binarização.
        kernel_blur_size (tuple): Tamanho do kernel para desfoque Gaussiano (e.g., (5,5)).
        kernel_morph_size (int): Tamanho do kernel para operações morfológicas (e.g., 3).
        debug_plots (bool): Se True, mostra plots para depuração.

    Returns:
        tuple: (mask_full, prominence_contour)
               Retorna (None, None) se nenhum contorno proeminente for encontrado.
    """
    # Desfoque Gaussiano
    blurred_diff = cv2.GaussianBlur(diferenca_raw, kernel_blur_size, 0)

    # Binarização
    _, thresh_diff = cv2.threshold(blurred_diff, threshold_value_difference, 255, cv2.THRESH_BINARY)

    # Operações Morfológicas (Open e Close)
    kernel = np.ones((kernel_morph_size, kernel_morph_size), np.uint8)
    mask_open = cv2.morphologyEx(thresh_diff, cv2.MORPH_OPEN, kernel)
    mask_full = cv2.morphologyEx(mask_open, cv2.MORPH_CLOSE, kernel)

    # Encontrar contornos
    contours, _ = cv2.findContours(mask_full, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    prominence_contour = None
    if contours:
        # Encontrar o maior contorno (assumindo que seja a gota mais proeminente)
        prominence_contour = max(contours, key=cv2.contourArea)

    if debug_plots:
        cv2.imshow("Processed Mask", mask_full)
        # cv2.waitKey(1)

    return mask_full, prominence_contour


def segment_drop(current_image_cv2, base_image_cv2, crop_coords,
                 threshold_value_difference=25, kernel_blur_size=(5, 5), kernel_morph_size=5, debug_plots=False):
    """
    Segmenta a gota da imagem usando subtração de fundo e processamento morfológico.

    Args:
        current_image_cv2 (np.array): Imagem atual (já rotacionada).
        base_image_cv2 (np.array): Imagem base (já rotacionada).
        crop_coords (list): [x1, y1, x2, y2] coordenadas de corte.
        threshold_value_difference (int): Limiar para a binarização da diferença.
        kernel_blur_size (tuple): Tamanho do kernel para o desfoque.
        kernel_morph_size (int): Tamanho do kernel para as operações morfológicas.
        debug_plots (bool): Se True, mostra plots para depuração.

    Returns:
        tuple: (mask_full, prominence_contour, cropped_current_color)
               Retorna dummy_mask/None/dummy_cropped_color se a gota "desaparecer" ou corte inválido.
    """
    diferenca_raw, cropped_current_color = calculate_image_difference(base_image_cv2, current_image_cv2, crop_coords, debug_plots)

    # If the droplet disappears or difference calculation failed due to invalid crop
    if diferenca_raw is None or cropped_current_color is None:
        # Get dimensions for dummy mask from crop_coords to match expected size
        x1, y1, x2, y2 = crop_coords
        dummy_mask_height = max(1, y2 - y1)
        dummy_mask_width = max(1, x2 - x1)
        
        dummy_mask = np.zeros((dummy_mask_height, dummy_mask_width), dtype=np.uint8)
        dummy_cropped_color = np.zeros((dummy_mask_height, dummy_mask_width, 3), dtype=np.uint8)
        return dummy_mask, None, dummy_cropped_color

    mask_full, prominence_contour = process_difference_image(
        diferenca_raw, threshold_value_difference, kernel_blur_size, kernel_morph_size, debug_plots
    )
    
    return mask_full, prominence_contour, cropped_current_color


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
# processing/image_processing.py
import cv2
import numpy as np
from datetime import datetime

def process_frame_for_droplets(current_frame, base_frame, frame_number, px_per_mm, mm3_per_ul):
    """
    Processa um único frame para detectar e medir gotas.

    Args:
        current_frame (np.array): A imagem do frame atual (já ajustada e cortada).
        base_frame (np.array): A imagem de fundo (já ajustada e cortada) para subtração.
        frame_number (int): O número do frame (para registro).
        px_per_mm (float): Pixels por milímetro para conversão de escala.
        mm3_per_ul (float): Fator de conversão de mm³ para µL.

    Returns:
        tuple: Uma tupla contendo:
            - np.array: O frame com as gotas detectadas e informações desenhadas.
            - dict: Dicionário contendo dados de medição para o frame.
                    Retorna None se nenhuma gota for detectada ou dados forem inválidos.
    """
    frame_data = {
        'frame_number': frame_number,
        'timestamp': datetime.now().isoformat(), # Usar datetime.now() ou um timestamp real se disponível
        'droplet_count': 0,
        'average_diameter_mm': 0.0,
        'average_volume_ul': 0.0
    }
    
    # --- Início da Lógica de Processamento de Imagem (Você precisará implementar isso) ---
    # Sugestão de passos:
    # 1. Subtração de fundo (background subtraction)
    # 2. Binarização (thresholding)
    # 3. Detecção de contornos (find contours)
    # 4. Filtragem de contornos (p.ex., por área, circularidade)
    # 5. Medição de propriedades das gotas (diâmetro, área)
    # 6. Desenhar resultados no frame
    
    # Exemplo de processamento BÁSICO (APENAS PARA FAZER FUNCIONAR)
    # Por favor, substitua isso pela sua lógica real de detecção de gotas.
    
    display_frame = current_frame.copy() # Frame para desenhar os resultados

    try:
        # Converta para escala de cinza
        gray_current = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        gray_base = cv2.cvtColor(base_frame, cv2.COLOR_BGR2GRAY)

        # Subtração de fundo
        diff = cv2.absdiff(gray_current, gray_base)
        
        # Binarização (ajuste o threshold conforme necessário)
        _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
        
        # Opcional: Operações morfológicas para limpar ruído
        kernel = np.ones((5,5),np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        # Detecção de contornos
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        droplet_diameters_px = []
        
        for contour in contours:
            # Filtrar por área para evitar pequenos ruídos
            area = cv2.contourArea(contour)
            if area < 50: # Ajuste este valor! Área mínima para uma gota (em pixels)
                continue
            
            # Ajuste de círculo para obter diâmetro
            (x,y), radius = cv2.minEnclosingCircle(contour)
            center = (int(x),int(y))
            radius = int(radius)
            
            # Desenhar o círculo e o centro (para depuração)
            cv2.circle(display_frame,center,radius,(0,255,0),2)
            # cv2.circle(display_frame,center,2,(0,0,255),3) # Centro
            
            diameter_px = 2 * radius
            droplet_diameters_px.append(diameter_px)

        frame_data['droplet_count'] = len(droplet_diameters_px)
        
        if droplet_diameters_px:
            avg_diameter_px = np.mean(droplet_diameters_px)
            avg_diameter_mm = avg_diameter_px / px_per_mm
            
            # Volume de uma esfera: V = (4/3) * pi * (diam/2)^3
            # Convertendo diâmetro de mm para raio em mm
            avg_radius_mm = avg_diameter_mm / 2.0
            avg_volume_mm3 = (4/3) * np.pi * (avg_radius_mm**3)
            avg_volume_ul = avg_volume_mm3 * mm3_per_ul # mm3_per_ul geralmente é 1 (1 mm³ = 1 µL)

            frame_data['average_diameter_mm'] = avg_diameter_mm
            frame_data['average_volume_ul'] = avg_volume_ul
            
            # Desenhar informações no frame
            info_text = f"Gotas: {frame_data['droplet_count']} | Diam: {avg_diameter_mm:.2f}mm | Vol: {avg_volume_ul:.2f}uL"
            cv2.putText(display_frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
        
    except Exception as e:
        print(f"Erro na lógica de processamento de gotas no frame {frame_number}: {e}")
        # Retorna o frame original ou um frame vazio em caso de erro
        return current_frame, None

    # --- Fim da Lógica de Processamento de Imagem ---

    return display_frame, frame_data
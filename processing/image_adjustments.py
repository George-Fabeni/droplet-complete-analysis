# processing/image_adjustments.py
import cv2
import numpy as np

def apply_adjustments_cv2(img_bgr, brightness_factor, exposure_factor, contrast_factor, highlights_factor, shadows_factor):
    """
    Applies image adjustments (brightness, exposure, contrast, highlights, shadows)
    using direct NumPy operations, which is compatible with OpenCV arrays.
    
    Args:
        img_bgr (np.array): Original image in BGR format (uint8).
        brightness_factor, exposure_factor, contrast_factor: All float, all multiplicative factors.
        highlights_factor: Multiplicative factor for pixels >= 128.
        shadows_factor (float): Multiplicative factor for pixels < 128.

    """
    
    # Conversão para float32 apenas uma vez
    img = img_bgr.astype(np.float32)
    
    # Ajuste de brilho e exposição (fatores multiplicativos combinados)
    img *= brightness_factor * exposure_factor
    
    # Ajuste de contraste em torno da média
    mean = np.mean(img, axis=(0, 1), keepdims=True)
    img = (img - mean) * contrast_factor + mean
    
    # Converte temporariamente para escala de cinza para gerar as máscaras
    gray = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    
    # Máscaras para sombras e luzes
    shadows_mask = gray < 128
    highlights_mask = ~shadows_mask  # mais rápido que gray >= 128
    
    # Aplicação direta nos canais RGB usando máscaras booleanas
    for c in range(3):  # B, G, R
        img[..., c][shadows_mask] *= shadows_factor
        img[..., c][highlights_mask] *= highlights_factor
    
    # Clipping final e conversão para uint8
    return np.clip(img, 0, 255).astype(np.uint8)


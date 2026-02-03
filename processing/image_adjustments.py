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

    img = img_bgr.astype(np.float32)
    
    # 1. Exposure (Multiplicativo)
    # Equação: $$I_{new} = I \times \text{exposure\_factor}$$
    img *= exposure_factor
    
    # 2. Brightness (Aditivo)
    # Equação: $$I_{new} = I + (100 \times \text{brightness\_factor} - 100)$$
    img += (100 * brightness_factor - 100)
    
    # 3. Contrast (Linear em torno da média)
    # Centralizamos na média para que o contraste não desloque o brilho geral
    mean = np.mean(img, axis=(0, 1), keepdims=True)
    img = (img - mean) * contrast_factor + mean

    # 4. Shadows & Highlights
    # Criamos a máscara baseada na luminância (Y) ou média simples
    # Usamos a imagem original ou a atual para definir o que é "sombra" ( < 128 )
    gray = cv2.cvtColor(np.clip(img, 0, 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
    shadows_mask = (gray < 128)[..., None] # Adiciona dimensão para broadcasting
    highlights_mask = ~shadows_mask
    
    # Aplicamos os fatores separadamente
    img = np.where(shadows_mask, img * shadows_factor, img)
    img = np.where(highlights_mask, img * highlights_factor, img)
    
    return np.clip(img, 0, 255).astype(np.uint8)


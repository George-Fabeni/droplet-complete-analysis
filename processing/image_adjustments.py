# processing/image_adjustments.py
import cv2
import numpy as np

def apply_adjustments_cv2(img_bgr, brightness_factor, exposure_factor, contrast_factor, highlights_factor, shadows_factor):
    """
    Applies image adjustments (brightness, exposure, contrast, highlights, shadows)
    using direct NumPy operations, which is compatible with OpenCV arrays.
    
    Args:
        img_bgr (np.array): Original image in BGR format (uint8).
        brightness_factor (float): Multiplicative factor for brightness (e.g., 1.2 for +20%).
        exposure_factor (float): Multiplicative factor for exposure (e.g., 1.2 for +20%).
        contrast_factor (float): Multiplicative factor for contrast (e.g., 1.2 for +20%).
        highlights_factor (float): Multiplicative factor for pixels >= 128.
        shadows_factor (float): Multiplicative factor for pixels < 128.

    Returns:
        np.array: Adjusted image in BGR format (uint8).
    """
    img_np = img_bgr.astype(np.float32)

    # Apply brightness
    img_np *= brightness_factor
    img_np = np.clip(img_np, 0, 255)

    # Apply exposure
    img_np *= exposure_factor
    img_np = np.clip(img_np, 0, 255)

    # Apply contrast around the mean
    mean = np.mean(img_np, axis=(0, 1), keepdims=True)
    img_np = (img_np - mean) * contrast_factor + mean
    img_np = np.clip(img_np, 0, 255)

    # Highlights and Shadows
    gray_temp = cv2.cvtColor(img_np.astype(np.uint8), cv2.COLOR_BGR2GRAY) # Convert to uint8 for cvtColor

    shadows_mask = (gray_temp < 128)
    highlights_mask = (gray_temp >= 128)

    shadows_mask_3d = shadows_mask[:, :, np.newaxis]
    highlights_mask_3d = highlights_mask[:, :, np.newaxis]

    img_np = np.where(shadows_mask_3d, img_np * shadows_factor, img_np)
    img_np = np.where(highlights_mask_3d, img_np * highlights_factor, img_np)
    
    img_np = np.clip(img_np, 0, 255).astype(np.uint8)

    return img_np
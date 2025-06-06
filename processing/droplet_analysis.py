# processing/droplet_analysis.py
import cv2
import numpy as np
import matplotlib.pyplot as plt
from config.settings import DEFAULT_THRESHOLD_VALUE_DIFFERENCE, DEFAULT_KERNEL_BLUR_SIZE, DEFAULT_KERNEL_MORPH_SIZE, \
                           DEFAULT_SYSTEM_THRESHOLD_VALUE, DEFAULT_SYSTEM_KERNEL_BLUR_SIZE, DEFAULT_SYSTEM_KERNEL_MORPH_SIZE

def segment_drop(current_frame_color, base_frame_color, crop_coords,
                 threshold_value_difference, kernel_blur_size, kernel_morph_size, debug_plots=False):
    """
    Segments the droplet by subtracting the base image.
    Returns the mask of the prominence and its contour.
    """
    if current_frame_color is None or base_frame_color is None:
        return None, None, None

    x1, y1, x2, y2 = crop_coords
    cropped_current_frame_color = current_frame_color[y1:y2, x1:x2].copy()
    cropped_base_frame_color = base_frame_color[y1:y2, x1:x2].copy()

    # Convert to grayscale
    gray_current = cv2.cvtColor(cropped_current_frame_color, cv2.COLOR_BGR2GRAY)
    gray_base = cv2.cvtColor(cropped_base_frame_color, cv2.COLOR_BGR2GRAY)

    # Calculate absolute difference
    diff = cv2.absdiff(gray_current, gray_base)

    # Apply blur
    diff_blurred = cv2.blur(diff, kernel_blur_size)

    # Threshold the difference image
    _, binary_diff = cv2.threshold(diff_blurred, threshold_value_difference, 255, cv2.THRESH_BINARY)

    # Apply morphological closing to fill gaps
    kernel = np.ones(kernel_morph_size, np.uint8)
    mask_prominence_closed = cv2.morphologyEx(binary_diff, cv2.MORPH_CLOSE, kernel)

    # Find contours on the closed mask
    contours, _ = cv2.findContours(mask_prominence_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    prominence_contour = None
    if contours:
        # Find the largest contour, assumed to be the droplet
        prominence_contour = max(contours, key=cv2.contourArea)

    if debug_plots:
        plt.figure("Debug - Segment Drop", figsize=(12, 6))
        plt.subplot(1, 4, 1)
        plt.imshow(cv2.cvtColor(cropped_current_frame_color, cv2.COLOR_BGR2RGB))
        plt.title("Current Cropped")
        plt.axis('off')

        plt.subplot(1, 4, 2)
        plt.imshow(diff_blurred, cmap='gray')
        plt.title("Difference (Blurred)")
        plt.axis('off')

        plt.subplot(1, 4, 3)
        plt.imshow(binary_diff, cmap='gray')
        plt.title("Thresholded Diff")
        plt.axis('off')

        plt.subplot(1, 4, 4)
        display_mask = cv2.cvtColor(mask_prominence_closed, cv2.COLOR_GRAY2BGR)
        if prominence_contour is not None:
            cv2.drawContours(display_mask, [prominence_contour], -1, (0, 255, 0), 2)
        plt.imshow(cv2.cvtColor(display_mask, cv2.COLOR_BGR2RGB))
        plt.title("Mask Prominence")
        plt.axis('off')
        plt.tight_layout()
        plt.show()

    return mask_prominence_closed, prominence_contour, cropped_current_frame_color

def get_system_contour(cropped_original_color, threshold_value, kernel_blur_size, kernel_morph_size, debug_plots=False):
    """
    Gets the contour of the droplet + substrate system from the original image (no subtraction).
    """
    if cropped_original_color is None:
        return None

    gray_cropped = cv2.cvtColor(cropped_original_color, cv2.COLOR_BGR2GRAY)
    
    imagem_borrada = cv2.blur(gray_cropped, kernel_blur_size)
    # THRESH_BINARY_INV assumes object is darker than background. Adjust if substrate is brighter.
    _, imagem_binaria = cv2.threshold(imagem_borrada, threshold_value, 255, cv2.THRESH_BINARY_INV)
    
    kernel = np.ones(kernel_morph_size, np.uint8)
    imagem_fechada = cv2.morphologyEx(imagem_binaria, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(imagem_fechada, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    system_contour = None
    if contours:
        system_contour = max(contours, key=cv2.contourArea)

    if debug_plots:
        plt.figure("Debug - Contorno do Sistema (Gota + Substrato)", figsize=(8, 8))
        display_system_contour = cv2.cvtColor(imagem_fechada, cv2.COLOR_GRAY2BGR)
        if system_contour is not None:
            cv2.drawContours(display_system_contour, [system_contour], -1, (0, 255, 0), 1)
        plt.imshow(cv2.cvtColor(display_system_contour, cv2.COLOR_BGR2RGB))
        plt.title("Contorno do Sistema (Gota + Substrato)")
        plt.axis('off')
        plt.tight_layout()
        plt.show()

    return system_contour

def calculate_measurements(mask_prominence_full, prominence_contour, px_per_mm, mm3_per_ul,
                           cropped_original_color=None, system_contour=None, debug_plots=False):
    """
    Calculates various droplet measurements (volume, base, height, contact angle).
    """
    measurements = {
        'volume_uL': 0,
        'base_length_pixels': 0,
        'base_length_mm': 0,
        'height_pixels': 0,
        'height_mm': 0,
        'cX': 0,
        'cY': 0,
        'form_factor': 0,
        'surface_area_air_mm2': 0,
        'surface_area_total_mm2': 0,
        'contact_angle_left_deg': 0,
        'contact_angle_right_deg': 0,
        'contact_angle_avg_deg': 0
    }

    if prominence_contour is None or prominence_contour.shape[0] == 0 or np.sum(mask_prominence_full) == 0:
        return measurements
    
    # --- Volume Calculation (using pixel count) ---
    area_pixels = np.sum(mask_prominence_full) / 255 # Sum of white pixels
    # Assuming droplet is a sphere cap or similar model
    # Volume calculation is complex and depends on droplet shape model.
    # For a simple approximation, convert area to volume:
    # A common (but rough) simplification for 2D area to 3D volume is Area * (sqrt(Area)/2)
    # Or, if assuming a spherical cap, it's a more involved formula.
    # Let's keep it based on Area for now, as it's a direct metric.
    # For a more accurate model, you'd need curve fitting for the profile.
    
    # Let's define a rough conversion from 2D pixel area to 3D volume (very simplified)
    # A better model for droplet volume involves fitting a curve and using the Young-Laplace equation.
    # For now, let's use a very basic approximation:
    # Assume 2D area (mask_prominence_full) is a proxy for volume.
    # This is a simplification and not physically accurate for all droplet shapes.
    # A more rigorous method would involve fitting a curve to the contour and calculating volume of revolution.
    
    # Volume calculation often uses models like spherical cap or Young-Laplace.
    # For simplification, let's re-use the area as a primary proxy
    # For a more physically accurate volume, you'd need the height and base and apply a cap formula.
    # A simpler proxy could be (Area_pixels)^(3/2) or similar, but the exact factor depends on calibration.
    # For this exercise, let's just convert pixel area to mm^2 then approximate volume.
    area_mm2 = area_pixels / (px_per_mm ** 2)
    
    # A very rough volume approximation from area (replace with proper model if needed)
    # If the drop is a hemisphere, Volume = (2/3) * pi * R^3, Area = pi * R^2
    # So, R = sqrt(Area/pi), Volume = (2/3) * Area * sqrt(Area/pi)
    # This is still a simplification, use actual methods for precise volume.
    # For now, let's just make it a direct conversion from area to a conceptual volume
    # This factor (0.1) is an arbitrary scaling for the example, adjust based on your actual drops
    measurements['volume_uL'] = area_mm2 * 0.1 / mm3_per_ul 
    
    # --- Centroid Calculation ---
    M = cv2.moments(mask_prominence_full)
    if M["m00"] != 0:
        measurements['cX'] = int(M["m10"] / M["m00"])
        measurements['cY'] = int(M["m01"] / M["m00"])

    # --- Height Calculation ---
    min_y_contour = np.min(prominence_contour[:, 0, 1])
    max_y_contour = np.max(prominence_contour[:, 0, 1])
    measurements['height_pixels'] = max_y_contour - min_y_contour
    measurements['height_mm'] = measurements['height_pixels'] / px_per_mm

    # --- Base Length Calculation and Contact Points ---
    base_y_coord = -1 # Coordenada Y da linha do substrato
    contact_point_left = None
    contact_point_right = None

    if system_contour is not None and system_contour.shape[0] > 0:
        base_y_coord = np.max(system_contour[:, 0, 1])
        
        # Filter points of prominence_contour that are close to base_y_coord
        tolerance = 2 # pixels
        prominence_y_at_base = prominence_contour[(prominence_contour[:, 0, 1] >= base_y_coord - tolerance) & 
                                                  (prominence_contour[:, 0, 1] <= base_y_coord + tolerance)]
        
        if prominence_y_at_base.size > 0:
            contact_point_left = prominence_y_at_base[np.argmin(prominence_y_at_base[:, 0])]
            contact_point_right = prominence_y_at_base[np.argmax(prominence_y_at_base[:, 0])]

            if contact_point_left is not None and contact_point_right is not None:
                measurements['base_length_pixels'] = np.linalg.norm(contact_point_left - contact_point_right)
                measurements['base_length_mm'] = measurements['base_length_pixels'] / px_per_mm
        else: # Fallback if no points found at base_y_coord
             # This can happen if crop is too tight or system_contour is not perfect.
             # If no points are exactly at base_y_coord, find the lowest points of the prominence contour
            lowest_y_prominence = np.max(prominence_contour[:, 0, 1])
            base_points_prominence = prominence_contour[prominence_contour[:, 0, 1] == lowest_y_prominence]
            if base_points_prominence.size > 0:
                contact_point_left = base_points_prominence[np.argmin(base_points_prominence[:, 0])]
                contact_point_right = base_points_prominence[np.argmax(base_points_prominence[:, 0])]
                measurements['base_length_pixels'] = np.linalg.norm(contact_point_left - contact_point_right)
                measurements['base_length_mm'] = measurements['base_length_pixels'] / px_per_mm

    # Fallback if contact points still not found
    if contact_point_left is None or contact_point_right is None:
        x_coords = prominence_contour[:, 0, 0]
        y_coords = prominence_contour[:, 0, 1]
        
        # Use min/max x and y of the whole contour if base detection failed for contact points
        # This is less accurate for >90 degree angles but provides a fallback.
        min_x_contour = np.min(x_coords)
        max_x_contour = np.max(x_coords)
        min_y_contour = np.min(y_coords)
        max_y_contour = np.max(y_coords)

        # For angle calculation fallback points, we need to pick actual points from the contour
        # This finds indices of the leftmost and rightmost points that are at the *lowest Y* of the contour
        # This is a heuristic that tries to get "base-like" points even without a clear substrate line.
        lowest_y_indices = np.where(y_coords == max_y_contour)[0]
        if lowest_y_indices.size > 0:
            fallback_left_idx = lowest_y_indices[np.argmin(x_coords[lowest_y_indices])]
            fallback_right_idx = lowest_y_indices[np.argmax(x_coords[lowest_y_indices])]
            contact_point_left = prominence_contour[fallback_left_idx, 0]
            contact_point_right = prominence_contour[fallback_right_idx, 0]
            measurements['base_length_pixels'] = np.linalg.norm(contact_point_left - contact_point_right)
            measurements['base_length_mm'] = measurements['base_length_pixels'] / px_per_mm


    # --- Form Factor Calculation (e.g., circularity for 2D) ---
    perimeter_pixels = cv2.arcLength(prominence_contour, True)
    if perimeter_pixels > 0:
        # Form Factor = 4 * PI * Area / Perimeter^2 (1 for perfect circle)
        measurements['form_factor'] = (4 * np.pi * area_pixels) / (perimeter_pixels ** 2)
    else:
        measurements['form_factor'] = 0

    # --- Surface Area Calculation (rough approximation for 3D) ---
    # This is a very rough approximation, typically needs a 3D model (e.g., spherical cap)
    # Area de superfície de uma calota esférica: 2 * PI * R * h_cap (R=raio da esfera, h_cap=altura da calota)
    # Ou usar a área 2D como proxy para a área de superfície (air)
    # Assumindo droplet is a segment of a sphere:
    # A more accurate calculation needs the droplet profile fitted to a curve.
    # For now, let's use a very basic heuristic:
    # Surface area (air) can be approximated from perimeter if assuming a cylindrical shape, or
    # from area if assuming spherical cap.
    # Using area_mm2 as a proxy for both air and total surface area.
    measurements['surface_area_air_mm2'] = area_mm2 # Placeholder
    measurements['surface_area_total_mm2'] = area_mm2 # Placeholder

    # --- Contact Angle Calculation ---
    calculate_contact_angle_new(prominence_contour, measurements, contact_point_left, contact_point_right, base_y_coord, 15, debug_plots, cropped_original_color)

    return measurements

def calculate_contact_angle_new(prominence_contour, measurements, contact_point_left, contact_point_right, base_y_coord, num_points_for_tangent=15, debug_plots=False, cropped_original_color=None):
    """
    Calculates contact angles based on detected contact points and contour.
    """
    if contact_point_left is None or contact_point_right is None or prominence_contour is None or prominence_contour.shape[0] < num_points_for_tangent * 2:
        measurements['contact_angle_left_deg'] = 0
        measurements['contact_angle_right_deg'] = 0
        measurements['contact_angle_avg_deg'] = 0
        return

    # Find the indices of the contact points in the full prominence contour
    idx_left_base = -1
    for i in range(prominence_contour.shape[0]):
        if np.isclose(prominence_contour[i, 0, 0], contact_point_left[0]) and np.isclose(prominence_contour[i, 0, 1], contact_point_left[1]):
            idx_left_base = i
            break
    
    idx_right_base = -1
    for i in range(prominence_contour.shape[0]):
        if np.isclose(prominence_contour[i, 0, 0], contact_point_right[0]) and np.isclose(prominence_contour[i, 0, 1], contact_point_right[1]):
            idx_right_base = i
            break
    
    if idx_left_base == -1 or idx_right_base == -1:
        measurements['contact_angle_left_deg'] = 0
        measurements['contact_angle_right_deg'] = 0
        measurements['contact_angle_avg_deg'] = 0
        return

    # --- Calculation for the LEFT contact angle ---
    points_for_fit_left = []
    for k in range(1, num_points_for_tangent + 1):
        idx = (idx_left_base + k) % prominence_contour.shape[0]
        points_for_fit_left.append(prominence_contour[idx, 0])
    points_for_fit_left = np.array(points_for_fit_left)

    angle_left_deg = 0
    if points_for_fit_left.shape[0] >= 2:
        # Calculate a robust tangent vector using linear regression over multiple points
        # Slope (m) from polyfit (y = mx + b)
        coeffs_left = np.polyfit(points_for_fit_left[:, 0], points_for_fit_left[:, 1], 1)
        m_left = coeffs_left[0] # Slope
        
        # Calculate vector for tangent: (1, m_left) if Y is up.
        # But here Y is down, so (1, m_left) where m_left is dy/dx.
        # Vector points from p_left_base towards avg_x_after_left, avg_y_after_left
        avg_x_left_segment = np.mean(points_for_fit_left[:, 0])
        avg_y_left_segment = np.mean(points_for_fit_left[:, 1])
        
        dx_left_vector = avg_x_left_segment - contact_point_left[0]
        dy_left_vector = avg_y_left_segment - contact_point_left[1]
        
        # Angle from positive X-axis in radians, then convert to degrees
        angle_from_x_axis_rad_left = np.arctan2(dy_left_vector, dx_left_vector)
        angle_from_x_axis_deg_left = np.degrees(angle_from_x_axis_rad_left)

        # Contact angle for left side: measured from horizontal line *into* the droplet.
        # If dy_left_vector is negative (line goes up), dx_left_vector is positive (line goes right)
        # -> angle_from_x_axis_deg_left will be negative. The contact angle is abs(this angle).
        # This handles angles < 90 and > 90 correctly.
        angle_left_deg = abs(angle_from_x_axis_deg_left)


    # --- Calculation for the RIGHT contact angle ---
    points_for_fit_right = []
    for k in range(1, num_points_for_tangent + 1):
        idx = (idx_right_base - k + prominence_contour.shape[0]) % prominence_contour.shape[0]
        points_for_fit_right.append(prominence_contour[idx, 0])
    points_for_fit_right = np.array(points_for_fit_right)

    angle_right_deg = 0
    if points_for_fit_right.shape[0] >= 2:
        coeffs_right = np.polyfit(points_for_fit_right[:, 0], points_for_fit_right[:, 1], 1)
        m_right = coeffs_right[0] # Slope

        avg_x_right_segment = np.mean(points_for_fit_right[:, 0])
        avg_y_right_segment = np.mean(points_for_fit_right[:, 1])

        # Vector from contact_point_right towards avg_x_right_segment, avg_y_right_segment
        dx_right_vector = avg_x_right_segment - contact_point_right[0]
        dy_right_vector = avg_y_right_segment - contact_point_right[1]

        # Angle from positive X-axis in radians, then convert to degrees
        angle_from_x_axis_rad_right = np.arctan2(dy_right_vector, dx_right_vector)
        angle_from_x_axis_deg_right = np.degrees(angle_from_x_axis_rad_right)

        # Contact angle for right side: measured from horizontal line *into* the droplet.
        # If dy_right_vector is negative (line goes up), dx_right_vector is negative (line goes left)
        # -> angle_from_x_axis_deg_right will be negative (e.g., -150).
        # The contact angle is 180 - abs(this angle).
        angle_right_deg = 180 - abs(angle_from_x_axis_deg_right)

    measurements['contact_angle_left_deg'] = angle_left_deg
    measurements['contact_angle_right_deg'] = angle_right_deg
    measurements['contact_angle_avg_deg'] = (angle_left_deg + angle_right_deg) / 2.0

    if debug_plots and cropped_original_color is not None:
        display_frame_debug = cropped_original_color.copy()
        cv2.drawContours(display_frame_debug, [prominence_contour], -1, (0, 255, 0), 1)
        
        if base_y_coord != -1:
            cv2.line(display_frame_debug, (0, base_y_coord), (display_frame_debug.shape[1], base_y_coord), (0, 0, 255), 1)
            cv2.putText(display_frame_debug, "Substrate Base", (5, base_y_coord - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

        if contact_point_left is not None:
            cv2.circle(display_frame_debug, tuple(contact_point_left.astype(int)), 5, (255, 0, 0), -1)
        if contact_point_right is not None:
            cv2.circle(display_frame_debug, tuple(contact_point_right.astype(int)), 5, (0, 0, 255), -1)

        if contact_point_left is not None and points_for_fit_left.shape[0] >= 2:
            tan_start_left = tuple(contact_point_left.astype(int))
            tan_end_left = (int(contact_point_left[0] + dx_left_vector * 10), int(contact_point_left[1] + dy_left_vector * 10))
            cv2.line(display_frame_debug, tan_start_left, tan_end_left, (255, 255, 0), 2)
            cv2.putText(display_frame_debug, f'{angle_left_deg:.1f} deg', (tan_end_left[0], tan_end_left[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA)

        if contact_point_right is not None and points_for_fit_right.shape[0] >= 2:
            tan_start_right = tuple(contact_point_right.astype(int))
            tan_end_right = (int(contact_point_right[0] + dx_right_vector * 10), int(contact_point_right[1] + dy_right_vector * 10))
            cv2.line(display_frame_debug, tan_start_right, tan_end_right, (0, 255, 255), 2)
            cv2.putText(display_frame_debug, f'{angle_right_deg:.1f} deg', (tan_end_right[0] - 50, tan_end_right[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

        plt.figure("Debug - Contact Angle Calculation", figsize=(8, 8))
        plt.imshow(cv2.cvtColor(display_frame_debug, cv2.COLOR_BGR2RGB))
        plt.title(f"Contact Angles: Left={angle_left_deg:.1f}°, Right={angle_right_deg:.1f}°")
        plt.axis('off')
        plt.tight_layout()
        plt.show()
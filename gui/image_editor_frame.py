# gui/image_editor_frame.py
import tkinter as tk
from tkinter import ttk
import cv2
from PIL import Image, ImageTk

from processing.image_adjustments import apply_adjustments_cv2
from config.settings import DEFAULT_BRIGHTNESS, DEFAULT_EXPOSURE, DEFAULT_CONTRAST, \
                           DEFAULT_HIGHLIGHTS, DEFAULT_SHADOWS, PREVIEW_THUMBNAIL_SIZE, \
                           THRESHOLD_VALUE_DIFFERENCE, KERNEL_BLUR_SIZE, KERNEL_MORPH_SIZE
from processing.image_processing import segment_drop

class ImageEditorFrame(ttk.Frame):
    def __init__(self, parent, crop_coords_ref, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self.parent = parent
        self.crop_coords_ref = crop_coords_ref
        
        self.current_image_to_adjust_cv2 = None # This will be the (potentially cropped) image
        self.base_image_for_display_cv2 = None  #To store the base (original/reference) image
        self.current_image_full_res_for_segment = None # NEW: Store full-res rotated current image
        self.base_image_full_res_for_segment = None   # NEW: Store full-res rotated base image

        self.display_second_image_tk = None # To hold the PhotoImage for the second label
        
        self.display_image_tk = None

        self.brightness_var = tk.IntVar(value=DEFAULT_BRIGHTNESS)
        self.exposure_var = tk.IntVar(value=DEFAULT_EXPOSURE)
        self.contrast_var = tk.IntVar(value=DEFAULT_CONTRAST)
        self.highlights_var = tk.IntVar(value=DEFAULT_HIGHLIGHTS)
        self.shadows_var = tk.IntVar(value=DEFAULT_SHADOWS)

        self._create_widgets()
        self._setup_layout()

    def _create_widgets(self):
        self.image_label = ttk.Label(self)
        self.image_label.grid(row=0, column=0, columnspan=1, padx=10, pady=10, sticky="nsew")
        
        self.second_image_label = ttk.Label(self)
        self.second_image_label.grid(row=0, column=1, columnspan=1, padx=2, pady=10, sticky="nsew")

        self.brightness_slider = tk.Scale(self, from_=0, to=200, orient=tk.HORIZONTAL, label="Brilho (%)", 
                                          variable=self.brightness_var, command=self._on_slider_change)
        self.exposure_slider = tk.Scale(self, from_=0, to=300, orient=tk.HORIZONTAL, label="Exposição (%)", 
                                        variable=self.exposure_var, command=self._on_slider_change)
        self.contrast_slider = tk.Scale(self, from_=0, to=300, orient=tk.HORIZONTAL, label="Contraste (%)", 
                                        variable=self.contrast_var, command=self._on_slider_change)
        self.highlights_slider = tk.Scale(self, from_=0, to=200, orient=tk.HORIZONTAL, label="Realces (%)", 
                                         variable=self.highlights_var, command=self._on_slider_change)
        self.shadows_slider = tk.Scale(self, from_=0, to=200, orient=tk.HORIZONTAL, label="Sombras (%)", 
                                       variable=self.shadows_var, command=self._on_slider_change)

    def _setup_layout(self):
        self.grid_rowconfigure(0, weight=1) # Makes the image row expand
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)

        self.brightness_slider.grid(row=1, column=0, columnspan=2, sticky='ew', padx=10, pady=2)
        self.exposure_slider.grid(row=2, column=0, columnspan=2, sticky='ew', padx=10, pady=2)
        self.contrast_slider.grid(row=3, column=0, columnspan=2, sticky='ew', padx=10, pady=2)
        self.highlights_slider.grid(row=4, column=0, columnspan=2, sticky='ew', padx=10, pady=2)
        self.shadows_slider.grid(row=5, column=0, columnspan=2, sticky='ew', padx=10, pady=2)

    def set_image(self, adjustable_image_cv2, base_image_display_cv2, current_full_res_for_segment, base_full_res_for_segment):
        """
        Sets the images for display and for processing.
        adjustable_image_cv2: The image (cropped/adjusted) for the left display.
        base_image_display_cv2: The image (original/base) for the right display, or None.
        current_full_res_for_segment: The full-res, rotated CURRENT image for segment_drop.
        base_full_res_for_segment: The full-res, rotated BASE image for segment_drop.
        """
        self.current_image_to_adjust_cv2 = adjustable_image_cv2.copy()
        # Handle case where base_image_display_cv2 might be None
        self.base_image_for_display_cv2 = base_image_display_cv2.copy() if base_image_display_cv2 is not None else None
        
        # Store the full-res images for processing in _update_display
        self.current_image_full_res_for_segment = current_full_res_for_segment.copy() if current_full_res_for_segment is not None else None
        self.base_image_full_res_for_segment = base_full_res_for_segment.copy() if base_full_res_for_segment is not None else None
        
        self._update_display()

    def _on_slider_change(self, val):
        self._update_display()

    def _update_display(self):
        if self.current_image_to_adjust_cv2 is None:
            return
        
        brightness = self.brightness_var.get() / 100.0
        exposure = self.exposure_var.get() / 100.0
        contrast = self.contrast_var.get() / 100.0
        highlights = self.highlights_var.get() / 100.0
        shadows = self.shadows_var.get() / 100.0

        adjusted_img = apply_adjustments_cv2(
            self.current_image_to_adjust_cv2, brightness, exposure, contrast, highlights, shadows
        )
        
        img_rgb = cv2.cvtColor(adjusted_img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        
        img_pil.thumbnail(PREVIEW_THUMBNAIL_SIZE, Image.LANCZOS)
        
        self.display_image_tk = ImageTk.PhotoImage(img_pil)
        self.image_label.config(image=self.display_image_tk)
        self.image_label.image = self.display_image_tk 
        
        # --- Logic for the SECOND (PROCESSED) image (right side) ---
        # Ensure we have the full-res images and crop coords to process
        if (self.current_image_to_adjust_cv2 is not None and 
            self.base_image_for_display_cv2 is not None and
            self.crop_coords_ref # This is the list from main_window
           ):
            
            # Apply adjustments to the full-res current image before sending to segment_drop
            # This is important if `segment_drop` relies on the adjusted appearance for thresholding.
            adjusted_current_cropped = apply_adjustments_cv2(
                self.current_image_to_adjust_cv2, brightness, exposure, contrast, highlights, shadows
            )

            # NOTE: The base_image_full_res_for_segment might also need adjustments if its thresholding
            # depends on the same lighting conditions as the current image.
            # For consistency, it's safer to apply adjustments to the base image before segmenting it too.
            adjusted_base_cropped = apply_adjustments_cv2(
                self.base_image_for_display_cv2, brightness, exposure, contrast, highlights, shadows
            )

            # Call segment_drop with the correctly typed (NumPy) images and the live crop_coords
            mask_full, prominence_contour, processed_image_display = segment_drop(
                adjusted_current_cropped, # Use the adjusted full-res current image
                adjusted_base_cropped,    # Use the adjusted full-res base image
                self.crop_coords_ref,      # Use the dynamically updated crop_coords from main_window
                THRESHOLD_VALUE_DIFFERENCE, KERNEL_BLUR_SIZE, KERNEL_MORPH_SIZE, debug_plots=False
            )
            
            # Now, `processed_image_display` contains the (cropped) result of segment_drop.
            # This is the image you want to show on the right.
            if processed_image_display is not None and processed_image_display.shape[0] > 0 and processed_image_display.shape[1] > 0:
                # Convert `processed_image_display` to RGB for PIL
                # `segment_drop` returns `cropped_current_color` which is BGR.
                second_img_rgb = cv2.cvtColor(processed_image_display, cv2.COLOR_BGR2RGB)
                
                # Optionally draw contour on the processed image
                if prominence_contour is not None and len(prominence_contour) > 0:
                    # Draw contour on a copy to not modify the original processed_image_display
                    processed_with_contour = second_img_rgb.copy()
                    # Ensure the contour is drawn in BGR format for cv2.drawContours if the input is BGR
                    # Or convert processed_with_contour back to BGR for drawing, then back to RGB
                    # Simpler: convert to BGR, draw, then back to RGB for PIL.
                    processed_with_contour_bgr = cv2.cvtColor(processed_with_contour, cv2.COLOR_RGB2BGR)
                    cv2.drawContours(processed_with_contour_bgr, [prominence_contour], -1, (0, 255, 0), 2) # Green contour
                    second_img_rgb = cv2.cvtColor(processed_with_contour_bgr, cv2.COLOR_BGR2RGB)


                second_img_pil = Image.fromarray(second_img_rgb)
                
                second_img_pil.thumbnail(PREVIEW_THUMBNAIL_SIZE, Image.LANCZOS)
                
                self.display_second_image_tk = ImageTk.PhotoImage(second_img_pil)
                self.second_image_label.config(image=self.display_second_image_tk)
                self.second_image_label.image = self.display_second_image_tk 
            else:
                # If segment_drop returns an empty or invalid image, clear the label
                self.second_image_label.config(image='')
                self.second_image_label.image = None
        else:
            # Clear the second image label if required full-res images are not available
            self.second_image_label.config(image='')
            self.second_image_label.image = None
            print("Falta alguma das condições")

    def get_adjustment_params(self):
        return {
            'brightness': self.brightness_var.get() / 100.0,
            'exposure': self.exposure_var.get() / 100.0,
            'contrast': self.contrast_var.get() / 100.0,
            'highlights': self.highlights_var.get() / 100.0,
            'shadows': self.shadows_var.get() / 100.0
        }

    def reset_adjustments(self):
        self.brightness_var.set(DEFAULT_BRIGHTNESS)
        self.exposure_var.set(DEFAULT_EXPOSURE)
        self.contrast_var.set(DEFAULT_CONTRAST)
        self.highlights_var.set(DEFAULT_HIGHLIGHTS)
        self.shadows_var.set(DEFAULT_SHADOWS)
        self._update_display()
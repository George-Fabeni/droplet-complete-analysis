# gui/image_editor_frame.py
import tkinter as tk
from tkinter import ttk
import cv2
from PIL import Image, ImageTk

from processing.image_adjustments import apply_adjustments_cv2
from config.settings import DEFAULT_BRIGHTNESS, DEFAULT_EXPOSURE, DEFAULT_CONTRAST, \
                           DEFAULT_HIGHLIGHTS, DEFAULT_SHADOWS, PREVIEW_THUMBNAIL_SIZE

class ImageEditorFrame(ttk.Frame):
    def __init__(self, parent, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self.parent = parent
        self.current_image_to_adjust_cv2 = None # This will be the (potentially cropped) image
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
        self.image_label.grid(row=0, column=0, columnspan=2, padx=10, pady=10, sticky="nsew")

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

    def set_image(self, img_cv2):
        """Sets the (potentially cropped) image to be edited and updates the display."""
        self.current_image_to_adjust_cv2 = img_cv2.copy()
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
        
        # Resize for preview display (maintain aspect ratio)
        img_pil.thumbnail(PREVIEW_THUMBNAIL_SIZE, Image.LANCZOS)
        
        self.display_image_tk = ImageTk.PhotoImage(img_pil)
        self.image_label.config(image=self.display_image_tk)
        self.image_label.image = self.display_image_tk

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
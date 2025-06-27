import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import cv2

# Certifique-se de que todas as importações necessárias estão aqui
from processing.image_adjustments import apply_adjustments_cv2
from processing.image_processing import segment_drop
from config.settings import DEFAULT_BRIGHTNESS, DEFAULT_EXPOSURE, DEFAULT_CONTRAST, DEFAULT_HIGHLIGHTS, DEFAULT_SHADOWS, \
                            PREVIEW_THUMBNAIL_SIZE, THRESHOLD_VALUE_DIFFERENCE, KERNEL_BLUR_SIZE, KERNEL_MORPH_SIZE # Import all necessary settings

class ImageEditorFrame(ttk.Frame):
    def __init__(self, parent, crop_coords_ref, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self.parent = parent
        self.crop_coords_ref = crop_coords_ref # Referência para as coordenadas de corte da janela principal

        # Estas imagens já virão CORTADAS da main_window
        self.current_image_to_adjust_cv2 = None
        self.base_image_for_display_cv2 = None

        # As imagens full-res para segment_drop não são mais necessárias aqui,
        # pois `segment_drop` agora receberá imagens já cortadas e ajustadas.
        # self.current_image_full_res_for_segment = None
        # self.base_image_full_res_for_segment = None

        self.display_image_tk = None
        self.display_second_image_tk = None

        self.brightness_var = tk.IntVar(value=DEFAULT_BRIGHTNESS)
        self.exposure_var = tk.IntVar(value=DEFAULT_EXPOSURE)
        self.contrast_var = tk.IntVar(value=DEFAULT_CONTRAST)
        self.highlights_var = tk.IntVar(value=DEFAULT_HIGHLIGHTS)
        self.shadows_var = tk.IntVar(value=DEFAULT_SHADOWS)

        self._create_widgets()
        self._setup_layout()

    def _create_widgets(self):
        self.image_label = ttk.Label(self)
        self.image_label.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

        self.second_image_label = ttk.Label(self)
        self.second_image_label.grid(row=0, column=1, padx=2, pady=10, sticky="nsew")

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

        self.brightness_slider.grid(row=1, column=0, columnspan=2, sticky='ew', padx=10, pady=2)
        self.exposure_slider.grid(row=2, column=0, columnspan=2, sticky='ew', padx=10, pady=2)
        self.contrast_slider.grid(row=3, column=0, columnspan=2, sticky='ew', padx=10, pady=2)
        self.highlights_slider.grid(row=4, column=0, columnspan=2, sticky='ew', padx=10, pady=2)
        self.shadows_slider.grid(row=5, column=0, columnspan=2, sticky='ew', padx=10, pady=2)

    def _setup_layout(self):
        self.grid_rowconfigure(0, weight=1) # Makes the image row expand
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)

    # Assinatura de set_image simplificada: recebe imagens JÁ CORTADAS
    def set_image(self, current_image_cropped, base_image_cropped):
        """
        Define as imagens a serem exibidas no editor.
        Ambas as imagens (current e base) devem vir JÁ CORTADAS da main_window.
        """
        self.current_image_to_adjust_cv2 = current_image_cropped.copy() if current_image_cropped is not None else None
        self.base_image_for_display_cv2 = base_image_cropped.copy() if base_image_cropped is not None else None

        # As variáveis full-res para segment_drop não são mais necessárias
        # self.current_image_full_res_for_segment = None
        # self.base_image_full_res_for_segment = None

        self._update_display() # Força a atualização da exibição

    def _on_slider_change(self, val):
        self._update_display()

    def _update_display(self):
        if self.current_image_to_adjust_cv2 is None:
            # Limpar ambas as labels se não houver imagem atual
            self.image_label.config(image='')
            self.image_label.image = None
            self.second_image_label.config(image='')
            self.second_image_label.image = None
            return

        brightness = self.brightness_var.get() / 100.0
        exposure = self.exposure_var.get() / 100.0
        contrast = self.contrast_var.get() / 100.0
        highlights = self.highlights_var.get() / 100.0
        shadows = self.shadows_var.get() / 100.0

        # Aplica ajustes à imagem atual JÁ CORTADA para o lado esquerdo
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
        # Garantir que ambas as imagens (atual e base) estejam disponíveis e cortadas
        if self.current_image_to_adjust_cv2 is not None and self.base_image_for_display_cv2 is not None:
            # Aplica ajustes à imagem atual e de base (ambas já cortadas) antes de enviar para segment_drop
            adjusted_current_for_segment = apply_adjustments_cv2(
                self.current_image_to_adjust_cv2, brightness, exposure, contrast, highlights, shadows
            )
            adjusted_base_for_segment = apply_adjustments_cv2(
                self.base_image_for_display_cv2, brightness, exposure, contrast, highlights, shadows
            )

            # Chama segment_drop com as imagens JÁ CORTADAS.
            # O crop_coords é passado, mas segment_drop não o usará para cortar, apenas para referência.
            mask_full, prominence_contour, processed_image_display = segment_drop(
                adjusted_current_for_segment, # Imagem atual JÁ CORTADA e ajustada
                adjusted_base_for_segment,    # Imagem de base JÁ CORTADA e ajustada
                self.crop_coords_ref,         # Passa as coords de corte (mas segment_drop não irá cortar novamente)
                THRESHOLD_VALUE_DIFFERENCE, KERNEL_BLUR_SIZE, KERNEL_MORPH_SIZE, debug_plots=False
            )

            if processed_image_display is not None and processed_image_display.shape[0] > 0 and processed_image_display.shape[1] > 0:
                second_img_rgb = cv2.cvtColor(processed_image_display, cv2.COLOR_BGR2RGB)

                if prominence_contour is not None and len(prominence_contour) > 0:
                    processed_with_contour_bgr = cv2.cvtColor(second_img_rgb, cv2.COLOR_RGB2BGR)
                    # O contorno já está nas coordenadas da imagem CROPPED, então desenha diretamente.
                    cv2.drawContours(processed_with_contour_bgr, [prominence_contour], -1, (0, 255, 0), 2) # Green contour
                    second_img_rgb = cv2.cvtColor(processed_with_contour_bgr, cv2.COLOR_BGR2RGB)

                second_img_pil = Image.fromarray(second_img_rgb)
                second_img_pil.thumbnail(PREVIEW_THUMBNAIL_SIZE, Image.LANCZOS)

                self.display_second_image_tk = ImageTk.PhotoImage(second_img_pil)
                self.second_image_label.config(image=self.display_second_image_tk)
                self.second_image_label.image = self.display_second_image_tk
            else:
                self.second_image_label.config(image='')
                self.second_image_label.image = None
        else:
            # Limpar a segunda label se a imagem de base não estiver disponível
            self.second_image_label.config(image='')
            self.second_image_label.image = None
            # print("Falta alguma das condições para exibir a imagem processada (base ou atual).")

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
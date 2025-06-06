# gui/main_window.py
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import os
import threading
import cv2
import numpy as np # Adicionar para operações com arrays

from processing.utils import load_image_cv2, get_image_paths_from_folder, load_image_from_dialog, rotate_image
from processing.video_generation import process_images_and_generate_video
from gui.image_editor_frame import ImageEditorFrame
from config.settings import DEFAULT_IMAGE_FOLDER, DEFAULT_OUTPUT_FOLDER, PX_PER_MM, MM3_PER_UL, PREVIEW_THUMBNAIL_SIZE, VIDEO_FPS

class DropletAnalyzerApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Analisador de Gotas")
        self.geometry("1200x800") # Adjust as needed

        self.image_paths = []
        self.current_image_index = -1
        self.current_image_full_res_cv2 = None # Store the full-resolution, rotated image
        self.current_image_display_cv2 = None # The image actually displayed in the GUI (cropped & resized)

        self.base_image_path = None
        self.rotation_angle = tk.DoubleVar(value=0.0)
        self.crop_coords = [0, 0, 0, 0] # [x1, y1, x2, y2]
        self.cropping_active = False
        self.start_x, self.start_y = -1, -1
        self.temp_crop_display_scaled = None # For faster drawing during cropping

        self._create_widgets()
        self._setup_layout()

        # Load images from default folder on startup
        self.load_images_from_folder(DEFAULT_IMAGE_FOLDER)

    def _create_widgets(self):
        left_panel = ttk.Frame(self, padding="10")
        left_panel.grid(row=0, column=0, sticky="nsew")
        left_panel.columnconfigure(0, weight=1)
        left_panel.rowconfigure(0, weight=0)
        left_panel.rowconfigure(1, weight=1)

        self.image_index_slider = tk.Scale(left_panel, from_=0, to=0, orient=tk.HORIZONTAL, 
                                             label="Imagem Atual", command=self._on_image_index_change)
        self.image_index_slider.grid(row=0, column=0, sticky='ew', padx=5, pady=5)

        self.image_editor_frame = ImageEditorFrame(left_panel)
        self.image_editor_frame.grid(row=1, column=0, sticky='nsew', pady=10)

        right_panel = ttk.Frame(self, padding="10")
        right_panel.grid(row=0, column=1, sticky="nsew")
        right_panel.columnconfigure(0, weight=1)

        ttk.Label(right_panel, text="Pasta de Imagens:").pack(pady=5)
        self.image_folder_path_var = tk.StringVar(value=DEFAULT_IMAGE_FOLDER)
        ttk.Entry(right_panel, textvariable=self.image_folder_path_var, width=50).pack(fill='x', padx=5, pady=2)
        ttk.Button(right_panel, text="Selecionar Pasta", command=self._select_image_folder).pack(fill='x', padx=5, pady=2)

        ttk.Label(right_panel, text="Imagem de Base:").pack(pady=5)
        self.base_image_path_var = tk.StringVar()
        ttk.Entry(right_panel, textvariable=self.base_image_path_var, width=50).pack(fill='x', padx=5, pady=2)
        ttk.Button(right_panel, text="Selecionar Imagem de Base", command=self._select_base_image).pack(fill='x', padx=5, pady=2)

        ttk.Label(right_panel, text="Ângulo de Rotação (°):").pack(pady=5)
        ttk.Entry(right_panel, textvariable=self.rotation_angle, width=10).pack(fill='x', padx=5, pady=2)
        self.rotation_angle.trace_add("write", self._on_rotation_angle_change) # Add trace to update image on angle change

        ttk.Label(right_panel, text="Coordenadas de Corte (X1,Y1,X2,Y2):").pack(pady=5)
        self.crop_coords_var = tk.StringVar(value="0,0,0,0")
        ttk.Entry(right_panel, textvariable=self.crop_coords_var, width=30).pack(fill='x', padx=5, pady=2)
        ttk.Button(right_panel, text="Definir Corte (na imagem visualizada)", command=self._start_cropping).pack(fill='x', padx=5, pady=2)

        ttk.Button(right_panel, text="Processar e Gerar Vídeo", command=self._start_processing).pack(pady=20, fill='x')

    def _setup_layout(self):
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=0)

    def load_images_from_folder(self, folder_path):
        self.image_paths = get_image_paths_from_folder(folder_path)
        if self.image_paths:
            self.image_index_slider.config(to=len(self.image_paths) - 1)
            self.current_image_index = 0
            self.image_index_slider.set(self.current_image_index)
            self._load_and_display_current_image()
        else:
            messagebox.showinfo("Info", "Nenhuma imagem encontrada na pasta selecionada.")
            self.image_index_slider.config(to=0)
            self.current_image_index = -1
            blank_image = np.zeros((PREVIEW_THUMBNAIL_SIZE[1], PREVIEW_THUMBNAIL_SIZE[0], 3), dtype=np.uint8)
            self.image_editor_frame.set_image(blank_image)

    def _select_image_folder(self):
        folder_selected = filedialog.askdirectory(initialdir=self.image_folder_path_var.get())
        if folder_selected:
            self.image_folder_path_var.set(folder_selected)
            self.load_images_from_folder(folder_selected)

    def _select_base_image(self):
        filepath = load_image_from_dialog()
        if filepath:
            self.base_image_path = filepath
            self.base_image_path_var.set(filepath)

    def _on_image_index_change(self, val):
        self.current_image_index = int(val)
        self._load_and_display_current_image()

    def _on_rotation_angle_change(self, *args):
        """Called when rotation angle changes, reloads and displays the image."""
        self._load_and_display_current_image()

    def _load_and_display_current_image(self):
        """
        Loads the full-res image, applies rotation, crops it if crop_coords are set,
        and then sends the result to the ImageEditorFrame for display.
        """
        if self.current_image_index != -1 and self.image_paths:
            current_path = self.image_paths[self.current_image_index]
            try:
                img_cv2 = load_image_cv2(current_path)
                
                # Store full-res rotated image
                self.current_image_full_res_cv2 = rotate_image(img_cv2, self.rotation_angle.get())
                
                # Apply cropping if crop_coords are defined and valid
                x1, y1, x2, y2 = self.crop_coords
                if x2 > x1 and y2 > y1 and \
                   x1 >= 0 and y1 >= 0 and \
                   x2 <= self.current_image_full_res_cv2.shape[1] and \
                   y2 <= self.current_image_full_res_cv2.shape[0]:
                    self.current_image_display_cv2 = self.current_image_full_res_cv2[y1:y2, x1:x2].copy()
                else:
                    self.current_image_display_cv2 = self.current_image_full_res_cv2.copy()

                # Send the (potentially cropped) image to the editor frame
                self.image_editor_frame.set_image(self.current_image_display_cv2)

            except Exception as e:
                messagebox.showerror("Erro de Imagem", f"Não foi possível carregar ou rotacionar a imagem: {e}")
                blank_image = np.zeros((PREVIEW_THUMBNAIL_SIZE[1], PREVIEW_THUMBNAIL_SIZE[0], 3), dtype=np.uint8)
                self.image_editor_frame.set_image(blank_image)
        else:
            blank_image = np.zeros((PREVIEW_THUMBNAIL_SIZE[1], PREVIEW_THUMBNAIL_SIZE[0], 3), dtype=np.uint8)
            self.image_editor_frame.set_image(blank_image)

    def _start_cropping(self):
        if self.current_image_full_res_cv2 is None:
            messagebox.showwarning("Aviso", "Nenhuma imagem carregada para corte.")
            return

        self.cropping_active = True
        
        # Create a scaled version of the full-res image ONCE for faster display during cropping
        h_orig, w_orig = self.current_image_full_res_cv2.shape[:2]
        display_w, display_h = PREVIEW_THUMBNAIL_SIZE
        self.scale_factor_crop_display = min(display_w / w_orig, display_h / h_orig)
        
        self.temp_crop_display_scaled = cv2.resize(self.current_image_full_res_cv2, 
                                                   (int(w_orig * self.scale_factor_crop_display), 
                                                    int(h_orig * self.scale_factor_crop_display)), 
                                                   interpolation=cv2.INTER_AREA)

        # Draw current crop coords if they exist on the scaled image
        x1, y1, x2, y2 = self.crop_coords
        if x2 > x1 and y2 > y1: # Check if valid crop exists
             cv2.rectangle(self.temp_crop_display_scaled, 
                           (int(x1 * self.scale_factor_crop_display), int(y1 * self.scale_factor_crop_display)), 
                           (int(x2 * self.scale_factor_crop_display), int(y2 * self.scale_factor_crop_display)), 
                           (0, 255, 0), 2)

        cv2.namedWindow("Selecione a Área de Corte (Arraste e Solte)", cv2.WINDOW_NORMAL)
        cv2.setWindowProperty("Selecione a Área de Corte (Arraste e Solte)", cv2.WND_PROP_TOPMOST, 1)

        # Set window size based on the scaled image
        cv2.resizeWindow("Selecione a Área de Corte (Arraste e Solte)", 
                         self.temp_crop_display_scaled.shape[1], self.temp_crop_display_scaled.shape[0])

        cv2.imshow("Selecione a Área de Corte (Arraste e Solte)", self.temp_crop_display_scaled)
        cv2.setMouseCallback("Selecione a Área de Corte (Arraste e Solte)", self._mouse_callback_crop)
        #messagebox.showinfo("Instruções de Corte", "Clique e arraste para selecionar a área de corte.\n"
        #                                         "Pressione 'Enter' para confirmar ou 'Esc' para cancelar.")
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == 13:  # Enter key
                break
            elif key == 27: # Esc key
                self.cropping_active = False
                # If cancelled, revert crop_coords to a default or previous valid state
                if all(c == 0 for c in self.crop_coords) or (self.crop_coords[2] <= self.crop_coords[0] or self.crop_coords[3] <= self.crop_coords[1]):
                    # If current crop is invalid, set to full image
                    h, w = self.current_image_full_res_cv2.shape[:2]
                    self.crop_coords = [0, 0, w, h]
                self.crop_coords_var.set(','.join(map(str, self.crop_coords)))
                break
        cv2.destroyWindow("Selecione a Área de Corte (Arraste e Solte)")
        self.cropping_active = False
        # Update the main display with the (possibly new) cropped image
        self._load_and_display_current_image()

    def _mouse_callback_crop(self, event, x, y, flags, param):
        if not self.cropping_active:
            return

        # Always start with a fresh copy of the scaled image for drawing
        display_img_for_cropping = self.temp_crop_display_scaled.copy()

        # Ensure 'x' and 'y' (mouse coordinates on the scaled display) are within bounds
        # These 'x' and 'y' are directly used for drawing on 'display_img_for_cropping'
        x = np.clip(x, 0, display_img_for_cropping.shape[1] - 1)
        y = np.clip(y, 0, display_img_for_cropping.shape[0] - 1)

        # Convert mouse coords (scaled) back to original image coords for storage in self.crop_coords
        h_orig, w_orig = self.current_image_full_res_cv2.shape[:2]
        x_orig = int(x / self.scale_factor_crop_display)
        y_orig = int(y / self.scale_factor_crop_display)
        
        # Ensure original coordinates are within original image bounds
        x_orig = np.clip(x_orig, 0, w_orig) # Use w_orig, h_orig directly for clipping max
        y_orig = np.clip(y_orig, 0, h_orig) # This is important to avoid off-by-one errors with image slicing

        if event == cv2.EVENT_LBUTTONDOWN:
            self.start_x, self.start_y = x_orig, y_orig
            # Initialize crop_coords with start point, end point will be updated on drag
            self.crop_coords = [self.start_x, self.start_y, x_orig, y_orig] 
            # Draw initial point to show user where click occurred
            cv2.circle(display_img_for_cropping, (x, y), 5, (0, 0, 255), -1) # Red dot for start
            cv2.imshow("Selecione a Área de Corte (Arraste e Solte)", display_img_for_cropping)

        elif event == cv2.EVENT_MOUSEMOVE and flags == cv2.EVENT_FLAG_LBUTTON:
            # Update the end point of crop_coords in original image scale
            self.crop_coords[2], self.crop_coords[3] = x_orig, y_orig
            
            # Draw on the scaled display using scaled start/end points
            # Get scaled start coordinates for drawing
            scaled_start_x = int(self.start_x * self.scale_factor_crop_display)
            scaled_start_y = int(self.start_y * self.scale_factor_crop_display)

            # Draw the rectangle using the scaled start point and current mouse position (x,y)
            cv2.rectangle(display_img_for_cropping, 
                          (scaled_start_x, scaled_start_y), 
                          (x, y), # Use current scaled mouse coords for the rectangle end point
                          (0, 255, 0), 2)
            cv2.imshow("Selecione a Área de Corte (Arraste e Solte)", display_img_for_cropping)

        elif event == cv2.EVENT_LBUTTONUP:
            # Final end point in original image scale
            self.crop_coords[2], self.crop_coords[3] = x_orig, y_orig
            
            # Ensure coords are (x1,y1,x2,y2) where x1<x2 and y1<y2
            self.crop_coords = [min(self.start_x, x_orig), min(self.start_y, y_orig), 
                                max(self.start_x, x_orig), max(self.start_y, y_orig)]
            
            # Update the Tkinter entry for display
            self.crop_coords_var.set(','.join(map(str, self.crop_coords)))
            
            # Draw final rectangle on the scaled display for visual confirmation
            # Use the finalized crop_coords (scaled) for drawing
            scaled_x1 = int(self.crop_coords[0] * self.scale_factor_crop_display)
            scaled_y1 = int(self.crop_coords[1] * self.scale_factor_crop_display)
            scaled_x2 = int(self.crop_coords[2] * self.scale_factor_crop_display)
            scaled_y2 = int(self.crop_coords[3] * self.scale_factor_crop_display)

            cv2.rectangle(display_img_for_cropping, 
                          (scaled_x1, scaled_y1), 
                          (scaled_x2, scaled_y2), 
                          (0, 255, 0), 2)
            cv2.imshow("Selecione a Área de Corte (Arraste e Solte)", display_img_for_cropping)



    def _start_processing(self):
        if not self.image_paths:
            messagebox.showwarning("Aviso", "Selecione uma pasta de imagens primeiro.")
            return
        if self.base_image_path is None:
            messagebox.showwarning("Aviso", "Selecione uma imagem de base primeiro.")
            return


        output_video_path = os.path.join(DEFAULT_OUTPUT_FOLDER, "processed_droplet_video.mp4")
        output_csv_path = os.path.join(DEFAULT_OUTPUT_FOLDER, "droplet_measurements.csv")

        os.makedirs(DEFAULT_OUTPUT_FOLDER, exist_ok=True)

        adj_params = self.image_editor_frame.get_adjustment_params()

        messagebox.showinfo("Processamento", "Iniciando processamento. Isso pode levar alguns minutos. Uma mensagem de conclusão aparecerá.")
        
        processing_thread = threading.Thread(target=self._run_processing, args=(
            self.image_paths, 
            self.base_image_path, 
            self.rotation_angle.get(), 
            self.crop_coords,
            output_video_path, 
            output_csv_path,
            adj_params['brightness'], adj_params['exposure'], adj_params['contrast'],
            adj_params['highlights'], adj_params['shadows']
        ))
        processing_thread.start()

    def _run_processing(self, *args):
        try:
            process_images_and_generate_video(*args)
            messagebox.showinfo("Sucesso", "Processamento e geração de vídeo concluídos!")
        except Exception as e:
            messagebox.showerror("Erro de Processamento", f"Ocorreu um erro durante o processamento: {e}")

    def on_closing(self):
        cv2.destroyAllWindows()
        self.destroy()

if __name__ == "__main__":
    app = DropletAnalyzerApp()
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()
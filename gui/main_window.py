import os
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import cv2
import numpy as np
import threading # Ensure this is imported

from processing.utils import load_image_cv2, get_image_paths_from_folder, load_image_from_dialog, rotate_image
from gui.image_editor_frame import ImageEditorFrame # Import ImageEditorFrame
from config.settings import DEFAULT_IMAGE_FOLDER, DEFAULT_OUTPUT_FOLDER, PREVIEW_THUMBNAIL_SIZE # Ensure PREVIEW_THUMBNAIL_SIZE is imported

class DropletAnalyzerApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Analisador de Gotas")
        self.geometry("1200x800")

        self.image_paths = []
        self.current_image_index = -1
        # self.current_image_full_res_cv2 = None # Não é mais necessário armazenar aqui
        self.current_image_display_cv2 = None # The image actually displayed in the GUI (cropped & resized)

        self.base_image_path_var = tk.StringVar() # Usar StringVar para o caminho da imagem de base
        self.rotation_angle = tk.DoubleVar(value=0.0)
        self.crop_coords = [0, 0, 0, 0] # [x1, y1, x2, y2]
        self.cropping_active = False
        self.start_x, self.start_y = -1, -1
        self.temp_crop_display_scaled = None # For faster drawing during cropping

        # NEW: Store full-res rotated, but UNCROPPED, images for passing to ImageEditorFrame
        self.current_image_full_res_uncropped = None
        self.base_image_full_res_uncropped = None # Carregado e rotacionado, mas sem corte

        self._create_widgets()
        self._setup_layout()

        # Load images from default folder on startup
        self.load_images_from_folder(DEFAULT_IMAGE_FOLDER)

    def _create_widgets(self):
        left_panel = ttk.Frame(self, padding="10")
        left_panel.grid(row=0, column=0, sticky="nsew")
        left_panel.columnconfigure(0, weight=1)
        left_panel.rowconfigure(1, weight=1) # ImageEditorFrame row

        self.image_index_slider = tk.Scale(left_panel, from_=0, to=0, orient=tk.HORIZONTAL,
                                            label="Imagem Atual", command=self._on_image_index_change)
        self.image_index_slider.grid(row=0, column=0, sticky='ew', padx=5, pady=5)

        # Pass the crop_coords directly as reference
        self.image_editor_frame = ImageEditorFrame(left_panel, self.crop_coords)
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
        self.grid_columnconfigure(1, weight=0) # Right panel should not expand horizontally

    def load_images_from_folder(self, folder_path):
        self.image_paths = get_image_paths_from_folder(folder_path)
        if self.image_paths:
            self.image_index_slider.config(to=len(self.image_paths) - 1)
            self.current_image_index = 0
            self.image_index_slider.set(self.current_image_index)
            self._load_and_display_current_image()
        else:
            self.image_index_slider.config(to=0)
            self.current_image_index = -1
            self.image_index_slider.set(0)
            self.image_editor_frame.set_image(None, None, None, None) # Clear images

    def _select_image_folder(self):
        folder_selected = filedialog.askdirectory(initialdir=self.image_folder_path_var.get())
        if folder_selected:
            self.image_folder_path_var.set(folder_selected)
            self.load_images_from_folder(folder_selected)

    def _select_base_image(self):
        filepath = load_image_from_dialog()
        if filepath:
            self.base_image_path_var.set(filepath)
            # Ao selecionar nova imagem de base, recarregar apenas ela e redesenhar tudo
            self._load_base_image_only()
            self._load_and_display_current_image() # Força atualização da GUI

    def _on_image_index_change(self, val):
        self.current_image_index = int(val)
        self._load_and_display_current_image()

    def _on_rotation_angle_change(self, *args):
        """Called when rotation angle changes, reloads and displays the image."""
        # Recarregar ambas as imagens (atual e base) com o novo ângulo de rotação
        self._load_base_image_only() # Recarrega a base com novo ângulo
        self._load_and_display_current_image() # Recarrega a atual com novo ângulo e redesenha

    def _load_base_image_only(self):
        """Carrega e rotaciona apenas a imagem de base, armazenando-a em full-res."""
        base_img_path_str = self.base_image_path_var.get()
        self.base_image_full_res_uncropped = None # Reset
        if base_img_path_str and os.path.exists(base_img_path_str):
            try:
                full_res_base = load_image_cv2(base_img_path_str)
                self.base_image_full_res_uncropped = rotate_image(full_res_base, self.rotation_angle.get())
                print(f"DEBUG: Base image reloaded (full-res, uncropped) for rotation angle change.")
            except Exception as base_load_error:
                messagebox.showwarning(
                    "Aviso de Imagem de Base",
                    f"Não foi possível carregar a imagem de base '{base_img_path_str}': {base_load_error}. "
                    "Continuando sem imagem de base para processamento."
                )
        else:
            print(f"DEBUG: Base image path empty or does not exist: '{base_img_path_str}'. No base image for processing.")


    def _load_and_display_current_image(self):
        """
        Carrega a imagem de resolução total, aplica rotação, corta-a se crop_coords estiverem definidos,
        e então envia o resultado para o ImageEditorFrame para exibição e processamento.
        """
        # Verifica se há imagens carregadas e um índice válido
        if self.current_image_index != -1 and self.image_paths:
            current_path = self.image_paths[self.current_image_index]

            # Inicializa variáveis
            img_cv2 = None
            current_image_full_res_rotated = None

            try:
                # 1. Carrega e rotaciona a imagem atual (full-res, ainda não cortada)
                img_full_res = load_image_cv2(current_path)
                current_image_full_res_rotated = rotate_image(img_full_res, self.rotation_angle.get())
                self.current_image_full_res_uncropped = current_image_full_res_rotated # Armazena a versão rotacionada completa

                # Se a imagem de base ainda não foi carregada (e rotacionada), faça-o agora.
                # Isso é crucial para que ela esteja disponível para o processamento.
                if self.base_image_full_res_uncropped is None and self.base_image_path_var.get():
                     self._load_base_image_only()

                # 2. Prepara a imagem para exibição e processamento, aplicando o corte AQUI.
                x1, y1, x2, y2 = self.crop_coords

                if current_image_full_res_rotated is not None:
                    h, w = current_image_full_res_rotated.shape[:2]
                    # Clamp the crop coordinates to the image dimensions
                    safe_x1 = max(0, min(x1, w))
                    safe_y1 = max(0, min(y1, h))
                    safe_x2 = max(0, min(x2, w))
                    safe_y2 = max(0, min(y2, h))

                    # Ensure valid cropping dimensions
                    if safe_x2 > safe_x1 and safe_y2 > safe_y1:
                        self.current_image_display_cv2 = current_image_full_res_rotated[safe_y1:safe_y2, safe_x1:safe_x2].copy()
                        # Também corta a imagem de base para exibição e processamento
                        if self.base_image_full_res_uncropped is not None:
                            base_image_cropped_for_display = self.base_image_full_res_uncropped[safe_y1:safe_y2, safe_x1:safe_x2].copy()
                        else:
                            base_image_cropped_for_display = None
                    else:
                        print("Warning: Invalid crop coordinates, displaying full image and base image (if available).")
                        self.current_image_display_cv2 = current_image_full_res_rotated.copy()
                        base_image_cropped_for_display = self.base_image_full_res_uncropped.copy() if self.base_image_full_res_uncropped is not None else None

                else:
                    print("Error: current_image_full_res_rotated is None.")
                    self.current_image_display_cv2 = np.zeros(PREVIEW_THUMBNAIL_SIZE + (3,), dtype=np.uint8) # Fallback blank image
                    base_image_cropped_for_display = None


                # 3. Envia as imagens cortadas para o ImageEditorFrame
                # A partir de agora, ImageEditorFrame e segment_drop trabalham com imagens JÁ CORTADAS.
                self.image_editor_frame.set_image(
                    self.current_image_display_cv2,            # Imagem atual CORTADA para o lado esquerdo
                    base_image_cropped_for_display,             # Imagem de base CORTADA para o lado direito
                    # Passamos None para os argumentos full_res porque o corte já foi feito
                    # e segment_drop vai receber as imagens já cortadas.
                    #None, # current_full_res_for_segment (não mais usado para segment_drop)
                    #None  # base_full_res_for_segment (não mais usado para segment_drop)
                )

            except Exception as e:
                messagebox.showerror("Erro de Carregamento/Processamento", f"Não foi possível carregar ou processar a imagem atual: {e}")
                # Exibe uma imagem em branco se houver erro ao carregar/processar a imagem atual
                blank_image = np.zeros(PREVIEW_THUMBNAIL_SIZE + (3,), dtype=np.uint8) # Cria uma imagem em branco colorida
                self.image_editor_frame.set_image(blank_image, None, None, None) # Passa None para todas as outras

        else:
            # Exibe uma imagem em branco se não houver imagens carregadas no folder
            blank_image = np.zeros(PREVIEW_THUMBNAIL_SIZE + (3,), dtype=np.uint8)
            self.image_editor_frame.set_image(blank_image, None, None, None) # Passa None para todas as outras

    def _start_cropping(self):
        # Usar a imagem completa e rotacionada para a seleção de corte
        if self.current_image_full_res_uncropped is None:
            messagebox.showwarning("Aviso", "Nenhuma imagem carregada para corte.")
            return

        self.cropping_active = True

        # Create a scaled version of the full-res image ONCE for faster display during cropping
        h_orig, w_orig = self.current_image_full_res_uncropped.shape[:2]
        display_w, display_h = PREVIEW_THUMBNAIL_SIZE
        self.scale_factor_crop_display = min(display_w / w_orig, display_h / h_orig)

        self.temp_crop_display_scaled = cv2.resize(self.current_image_full_res_uncropped,
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

        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == 13:  # Enter key
                break
            elif key == 27: # Esc key
                self.cropping_active = False
                # If cancelled, revert crop_coords to a default or previous valid state
                if all(c == 0 for c in self.crop_coords) or (self.crop_coords[2] <= self.crop_coords[0] or self.crop_coords[3] <= self.crop_coords[1]):
                    # If current crop is invalid, set to full image
                    h, w = self.current_image_full_res_uncropped.shape[:2]
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

        display_img_for_cropping = self.temp_crop_display_scaled.copy()

        x = np.clip(x, 0, display_img_for_cropping.shape[1] - 1)
        y = np.clip(y, 0, display_img_for_cropping.shape[0] - 1)

        # Convert mouse coords (scaled) back to original image coords for storage in self.crop_coords
        h_orig, w_orig = self.current_image_full_res_uncropped.shape[:2]
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
            scaled_x1 = int(min(self.start_x, x_orig) * self.scale_factor_crop_display)
            scaled_y1 = int(min(self.start_y, y_orig) * self.scale_factor_crop_display)
            scaled_x2 = int(max(self.start_x, x_orig) * self.scale_factor_crop_display)
            scaled_y2 = int(max(self.start_y, y_orig) * self.scale_factor_crop_display)


            # Draw the rectangle using the scaled start point and current mouse position (x,y)
            cv2.rectangle(display_img_for_cropping,
                          (scaled_x1, scaled_y1),
                          (scaled_x2, scaled_y2), # Use current scaled mouse coords for the rectangle end point
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
        # A validação da imagem de base agora usa base_image_full_res_uncropped
        if self.base_image_full_res_uncropped is None:
            messagebox.showwarning("Aviso", "Selecione e carregue uma imagem de base válida primeiro.")
            return


        output_video_path = os.path.join(DEFAULT_OUTPUT_FOLDER, "processed_droplet_video.mp4")
        output_csv_path = os.path.join(DEFAULT_OUTPUT_FOLDER, "droplet_measurements.csv")

        os.makedirs(DEFAULT_OUTPUT_FOLDER, exist_ok=True)

        adj_params = self.image_editor_frame.get_adjustment_params()

        messagebox.showinfo("Processamento", "Iniciando processamento. Isso pode levar alguns minutos. Uma mensagem de conclusão aparecerá.")

        # Passamos as imagens FULL-RES ROTACIONADAS para o processamento em lote
        processing_thread = threading.Thread(target=self._run_processing, args=(
            self.image_paths,
            self.base_image_full_res_uncropped, # Passa a imagem de base full-res já rotacionada
            self.rotation_angle.get(), # Ângulo de rotação é para o processamento
            self.crop_coords,
            output_video_path,
            output_csv_path,
            adj_params['brightness'], adj_params['exposure'], adj_params['contrast'],
            adj_params['highlights'], adj_params['shadows']
        ))
        processing_thread.start()

    def _run_processing(self, *args):
        try:
            # process_images_and_generate_video agora espera a imagem de base já rotacionada
            # e os crop_coords serão aplicados dentro dela, junto com os ajustes.
            from processing.image_processing import process_images_and_generate_video # Ensure this import is here
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
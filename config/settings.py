# config/settings.py

# Parâmetros de calibração (ajuste conforme seu setup)
PX_PER_MM = 163.28  # Exemplo: 20 pixels por milímetro
MM3_PER_UL = 1.0  # 1 mm^3 = 1 uL (conversão padrão)

# Caminhos padrão
DEFAULT_IMAGE_FOLDER = "images/" # Onde suas imagens de entrada estarão
DEFAULT_OUTPUT_FOLDER = "output/" # Onde o vídeo e resultados serão salvos

# Parâmetros padrão para processamento de imagem
DEFAULT_THRESHOLD_VALUE_DIFFERENCE = 30
DEFAULT_KERNEL_BLUR_SIZE = (5, 5)
DEFAULT_KERNEL_MORPH_SIZE = (5, 5)

# Parâmetros padrão para a detecção da base (para system_contour)
DEFAULT_SYSTEM_THRESHOLD_VALUE = 80 # Pode ser diferente da diferença para pegar o objeto e o substrato
DEFAULT_SYSTEM_KERNEL_BLUR_SIZE = (3, 3)
DEFAULT_SYSTEM_KERNEL_MORPH_SIZE = (3, 3)

# Parâmetros padrão dos sliders (100 = 1.0x, sem alteração)
DEFAULT_BRIGHTNESS = 100
DEFAULT_EXPOSURE = 100
DEFAULT_CONTRAST = 100
DEFAULT_HIGHLIGHTS = 100
DEFAULT_SHADOWS = 100

# Outras configurações
DEBUG_PLOTS_ENABLED = False # Habilita plots de depuração em janelas separadas
PREVIEW_THUMBNAIL_SIZE = (600, 400) # Tamanho máximo para a imagem de preview na GUI
VIDEO_FPS = 10.0 # Frames por segundo do vídeo de saída

THRESHOLD_VALUE_DIFFERENCE = 25 # Threshold para a diferença de imagem
KERNEL_BLUR_SIZE = (5, 5) # Tamanho do kernel para o desfoque Gaussiano
KERNEL_MORPH_SIZE = 5 # Tamanho do kernel para operações morfológicas (OPEN/CLOSE)

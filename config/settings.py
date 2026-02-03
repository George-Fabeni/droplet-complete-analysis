# config/settings.py

# Calibration parameters
PX_PER_MM = 167  # Exemple: 20 pixels per millimeter
MM3_PER_UL = 1.0  # 1 mm^3 = 1 uL (standart conversion)
INITIAL_CONC = 6

# Standart paths
DEFAULT_IMAGE_FOLDER = "images" # Input images location
DEFAULT_OUTPUT_FOLDER = "output" # Output video location

# Default slider values (100 = 1.0x)
DEFAULT_BRIGHTNESS = 100
DEFAULT_EXPOSURE = 100
DEFAULT_CONTRAST = 100
DEFAULT_HIGHLIGHTS = 100
DEFAULT_SHADOWS = 100

# Other configs

PREVIEW_THUMBNAIL_SIZE = (450, 300) # Max size for image preview in GUI
VIDEO_FPS = 10.0 # FPS of output video

THRESHOLD_VALUE_DIFFERENCE = 60 # Threshold for image difference
KERNEL_BLUR_SIZE = (5, 5) # Kernel size for gaussian blur
KERNEL_MORPH_SIZE = 5 # Kernel size for morphological operations

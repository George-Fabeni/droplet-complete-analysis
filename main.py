# main.py
import os
import sys

# Adiciona o diretório raiz do projeto ao PATH para que os módulos possam ser importados
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from gui.main_window import DropletAnalyzerApp

if __name__ == "__main__":
    app = DropletAnalyzerApp()
    app.mainloop()
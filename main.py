# main.py
import os
import sys

# Adiciona o diretório raiz do projeto ao PATH para que os módulos possam ser importados
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from gui.main_window import DropletAnalyzerApp

def main():
    try:
        app = DropletAnalyzerApp()
        app.protocol("WM_DELETE_WINDOW", app.on_closing)
        app.mainloop()
    except Exception as e:
        print(f"App error: {e}")
    finally:
        try:
            app.quit()
        except:
            pass

if __name__ == "__main__":
    main()
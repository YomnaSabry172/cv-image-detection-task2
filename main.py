"""Entry point: launches the CV Task 2 desktop app (owner: Lead)."""
import sys

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication

from ui import MainWindow, apply_theme


# Creates the Qt application, applies the dark theme and opens the main window.
def main() -> None:
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    app = QApplication(sys.argv)
    apply_theme(app)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

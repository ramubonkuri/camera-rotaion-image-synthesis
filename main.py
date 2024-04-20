import cv2
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QLabel, QFileDialog, QInputDialog
from external_depth_model import ExternalDepthModel

class DynamicViewSynthesis:
    def __init__(self):
        self.model = ExternalDepthModel()

    def load_image(self, filepath):
        return cv2.imread(filepath)

    def open_and_transform_image(self, filepath):
        image = self.load_image(filepath)
        newPositionX, newPositionY = 0.2, 0.2  # Example values, replace with user input or other sources as needed
        new_position = [newPositionX, newPositionY, 0]  # Adding '0' for Z-axis for completeness
        
        point_cloud = self.model.change_viewpoint(image, new_position)
        #self.model.visualize_point_cloud(point_cloud)
        cv2.imshow("Synthesized Image", point_cloud)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def main_gui():
    app = QApplication([])
    window = QWidget()
    layout = QVBoxLayout()
    title = QLabel('Dynamic View Synthesis Application')
    open_button = QPushButton('Open Fisheye Image')
    synthesis = DynamicViewSynthesis()

    def open_image():
        filepath, _ = QFileDialog.getOpenFileName()
        if filepath:
            synthesis.open_and_transform_image(filepath)
        
    open_button.clicked.connect(open_image)
    layout.addWidget(title)
    layout.addWidget(open_button)
    window.setLayout(layout)
    window.show()
    app.exec_()

if __name__ == "__main__":
    main_gui()

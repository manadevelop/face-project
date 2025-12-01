import sys
import cv2
import numpy as np
import requests
from PyQt5 import QtWidgets, QtGui, QtCore
import mediapipe as mp

# Cambia esto si tu backend corre en otra IP/puerto
BACKEND_URL = "http://127.0.0.1:8000"


class FaceClient(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Face Client - Captura y Enrolamiento")
        self.setGeometry(100, 100, 900, 600)

        # ---------- Zona de video ----------
        self.video_label = QtWidgets.QLabel()
        self.video_label.setFixedSize(640, 480)
        self.video_label.setStyleSheet("background-color: #000;")

        # ---------- Controles ----------
        self.mode_combo = QtWidgets.QComboBox()
        self.mode_combo.addItems(["ENROLAR", "VERIFICAR 1:1", "IDENTIFICAR 1:N"])

        self.name_input = QtWidgets.QLineEdit()
        self.name_input.setPlaceholderText("Nombre / ID persona (para enrolar o verificación 1:1)")

        self.capture_button = QtWidgets.QPushButton("Capturar y enviar")
        self.capture_button.clicked.connect(self.on_capture)

        self.stop_button = QtWidgets.QPushButton("Detener cámara")
        self.stop_button.clicked.connect(self.on_toggle_camera)

        self.status_label = QtWidgets.QLabel("Estado: listo")

        controls_layout = QtWidgets.QVBoxLayout()
        controls_layout.addWidget(QtWidgets.QLabel("Modo:"))
        controls_layout.addWidget(self.mode_combo)
        controls_layout.addSpacing(10)
        controls_layout.addWidget(QtWidgets.QLabel("Nombre / ID:"))
        controls_layout.addWidget(self.name_input)
        controls_layout.addSpacing(20)
        controls_layout.addWidget(self.capture_button)
        controls_layout.addWidget(self.stop_button)
        controls_layout.addStretch()
        controls_layout.addWidget(self.status_label)

        right_widget = QtWidgets.QWidget()
        right_widget.setLayout(controls_layout)

        main_layout = QtWidgets.QHBoxLayout()
        main_layout.addWidget(self.video_label)
        main_layout.addWidget(right_widget)

        central_widget = QtWidgets.QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

        # ---------- Cámara ----------
        self.cap = cv2.VideoCapture(0)
        self.camera_running = self.cap.isOpened()
        if not self.camera_running:
            self.status_label.setText("Estado: NO se pudo abrir la cámara. Revisa permisos en macOS.")
            self.show_info_screen()
        else:
            self.status_label.setText("Estado: cámara iniciada.")

        # ---------- MediaPipe ----------
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils

        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=0,
            min_detection_confidence=0.5
        )

        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # Último rostro recortado listo para enviar al backend
        self.current_face_roi = None

        # Timer para refrescar video
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_frame)
        if self.camera_running:
            self.timer.start(30)  # ~33 fps

    # ---------- Gestión de ventana ----------

    def closeEvent(self, event):
        """Liberar la cámara al cerrar la ventana."""
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()
        super().closeEvent(event)

    # ---------- Cámara y dibujado ----------

    def update_frame(self):
        """Captura un frame de la cámara, corre MediaPipe y dibuja resultados."""
        if not self.camera_running or self.cap is None or not self.cap.isOpened():
            return

        ret, frame = self.cap.read()
        if not ret:
            return

        # OpenCV: BGR -> RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, _ = frame_rgb.shape

        # Detección de rostro
        results_det = self.face_detection.process(frame_rgb)
        self.current_face_roi = None

        if results_det.detections:
            for detection in results_det.detections:
                bbox = detection.location_data.relative_bounding_box
                x_min = int(bbox.xmin * w)
                y_min = int(bbox.ymin * h)
                box_w = int(bbox.width * w)
                box_h = int(bbox.height * h)

                x_min = max(0, x_min)
                y_min = max(0, y_min)
                x_max = min(w, x_min + box_w)
                y_max = min(h, y_min + box_h)

                # Recorte para enviar al backend (RGB)
                self.current_face_roi = frame_rgb[y_min:y_max, x_min:x_max].copy()

                # Face Mesh solo para visualización
                face_roi_for_mesh = frame_rgb[y_min:y_max, x_min:x_max].copy()
                face_mesh_results = self.face_mesh.process(face_roi_for_mesh)

                if face_mesh_results.multi_face_landmarks:
                    for face_landmarks in face_mesh_results.multi_face_landmarks:
                        self.mp_drawing.draw_landmarks(
                            image=face_roi_for_mesh,
                            landmark_list=face_landmarks,
                            connections=self.mp_face_mesh.FACEMESH_TESSELATION,
                            landmark_drawing_spec=None,
                            connection_drawing_spec=self.mp_drawing.DrawingSpec(
                                color=(0, 255, 255),
                                thickness=1,
                                circle_radius=1
                            )
                        )

                # Reemplazar ROI original por el que tiene los landmarks dibujados
                frame_rgb[y_min:y_max, x_min:x_max] = face_roi_for_mesh

                # Dibujar bounding box
                cv2.rectangle(frame_rgb, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

                # Solo usamos el primer rostro detectado
                break

        # Mostrar frame en el QLabel
        image = QtGui.QImage(
            frame_rgb.data,
            frame_rgb.shape[1],
            frame_rgb.shape[0],
            frame_rgb.strides[0],
            QtGui.QImage.Format_RGB888
        )
        pixmap = QtGui.QPixmap.fromImage(image)
        self.video_label.setPixmap(pixmap)

    def show_info_screen(self):
        """Muestra una pantalla blanca con info del proyecto."""
        pixmap = QtGui.QPixmap(self.video_label.width(), self.video_label.height())
        pixmap.fill(QtGui.QColor("white"))

        painter = QtGui.QPainter(pixmap)
        painter.setPen(QtGui.QPen(QtGui.QColor("black")))

        # Título del proyecto
        font = painter.font()
        font.setPointSize(18)
        font.setBold(True)
        painter.setFont(font)
        painter.drawText(
            QtCore.QRect(0, 40, pixmap.width(), 40),
            QtCore.Qt.AlignHCenter,
            "Proyecto: Identificación facial"
        )

        # Integrantes
        font.setPointSize(12)
        font.setBold(False)
        painter.setFont(font)
        integrantes_text = "Integrantes:\n " \
        "- Kuan Becerra, Orlando José\n " \
        "- León Pacheco, Alex Celestino\n " \
        "- Minchán Ramos, Edwin Jhon\n " \
        "- Montes Jaramillo, Victor Fernando\n " \
        "- Marco Antonio, Nina Aguilar"
        painter.drawText(
            QtCore.QRect(40, 120, pixmap.width() - 100, 120),
            QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop,
            integrantes_text
        )

        # Descripción breve
        desc_text = (
            "Descripción:\n"
            "Sistema de enrolamiento y verificación biométrica facial\n"
            "usando MediaPipe en el cliente y DeepFace en el backend."
        )
        painter.drawText(
            QtCore.QRect(40, 220, pixmap.width() - 80, 120),
            QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop,
            desc_text
        )

        painter.end()
        self.video_label.setPixmap(pixmap)

    def on_toggle_camera(self):
        """Detiene o reanuda la cámara y muestra/oculta la pantalla info."""
        if self.camera_running:
            # Detener
            self.camera_running = False
            self.timer.stop()
            if self.cap is not None and self.cap.isOpened():
                self.cap.release()
            self.cap = None
            self.stop_button.setText("Reanudar cámara")
            self.status_label.setText("Estado: cámara detenida.")
            self.show_info_screen()
        else:
            # Reanudar
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                self.status_label.setText("Estado: NO se pudo reanudar la cámara.")
                self.show_info_screen()
                return
            self.camera_running = True
            self.timer.start(30)
            self.stop_button.setText("Detener cámara")
            self.status_label.setText("Estado: cámara reanudada.")

    # ---------- Lógica de captura / envío ----------

    def on_capture(self):
        """Captura el rostro actual y lo envía al backend según el modo."""
        mode = self.mode_combo.currentText()
        name = self.name_input.text().strip()

        if self.current_face_roi is None:
            self.status_label.setText("Estado: no se detectó rostro para capturar.")
            return

        # current_face_roi está en RGB -> convertir a BGR para codificar JPG
        face_bgr = cv2.cvtColor(self.current_face_roi, cv2.COLOR_RGB2BGR)
        ok, buf = cv2.imencode(".jpg", face_bgr)
        if not ok:
            self.status_label.setText("Estado: error al codificar rostro.")
            return

        files = {
            "image": ("face.jpg", buf.tobytes(), "image/jpeg")
        }

        try:
            if mode == "ENROLAR":
                if not name:
                    self.status_label.setText("Estado: ingresa Nombre/ID para enrolar.")
                    return

                data = {
                    "person_id": name,
                    "name": name,
                }
                resp = requests.post(f"{BACKEND_URL}/enroll", data=data, files=files, timeout=20)
                resp.raise_for_status()
                j = resp.json()
                msg = j.get("message", "Enrolamiento OK")
                total = j.get("total_embeddings", "?")
                self.status_label.setText(f"Estado: {msg} (total={total})")

            elif mode == "VERIFICAR 1:1":
                if not name:
                    self.status_label.setText("Estado: ingresa Nombre/ID esperado para 1:1.")
                    return

                resp = requests.post(f"{BACKEND_URL}/identify", files=files, timeout=20)
                resp.raise_for_status()
                j = resp.json()
                best_id = j.get("person_id") or j.get("name")
                sim = j.get("similarity", 0.0)

                if best_id is None:
                    self.status_label.setText("Estado: sin coincidencias en la base.")
                elif best_id == name:
                    self.status_label.setText(
                        f"Estado: VERIFICADO 1:1 como {best_id} (sim={sim:.2f})"
                    )
                else:
                    self.status_label.setText(
                        f"Estado: NO corresponde al ID {name}. Mejor match={best_id} (sim={sim:.2f})"
                    )

            elif mode == "IDENTIFICAR 1:N":
                resp = requests.post(f"{BACKEND_URL}/identify", files=files, timeout=20)
                resp.raise_for_status()
                j = resp.json()
                best_name = j.get("name")
                sim = j.get("similarity")

                if best_name is None:
                    self.status_label.setText("Estado: sin coincidencias en la base.")
                else:
                    self.status_label.setText(
                        f"Estado: mejor match = {best_name} (sim={sim:.2f})"
                    )

        except requests.RequestException as e:
            self.status_label.setText(f"Estado: error al llamar al backend: {e}")


def main():
    app = QtWidgets.QApplication(sys.argv)
    window = FaceClient()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

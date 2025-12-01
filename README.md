# Face Project (Face Backend + Face Client)

Este repositorio contiene un proyecto completo de demostración de **reconocimiento facial** compuesto por dos componentes principales:

- **Face Backend (`face_backend`)**  
  API REST construida con **FastAPI** y **DeepFace** que:
  - Genera embeddings faciales usando el modelo ArcFace.
  - Enrola personas (almacena embeddings).
  - Realiza identificación 1:N dado un rostro de entrada.

- **Face Client (`face_client`)**  
  Aplicación de escritorio en **PyQt5** que:
  - Captura video desde la cámara usando OpenCV.
  - Detecta rostros y dibuja la malla facial con MediaPipe (Face Detection + Face Mesh).
  - Recorta el rostro y lo envía al backend para enrolar, verificar 1:1 o identificar 1:N.

---

## 1. Estructura del repositorio

```text
FACE_PROJECT/
├─ face_backend/
│  ├─ main.py           # API FastAPI + DeepFace (endpoints /enroll, /identify, /health)
│  ├─ models/
│  │  └─ embeddings.pkl # Base de embeddings (NO se versiona)
│  ├─ venv/             # Entorno virtual del backend (NO se versiona)
│  └─ README.md         # Documentación específica del backend
│
├─ face_client/
│  ├─ client_gui.py     # Cliente de escritorio PyQt5 + OpenCV + MediaPipe
│  ├─ venv/             # Entorno virtual del cliente (NO se versiona)
│  └─ README.md         # Documentación específica del cliente
│
├─ .gitignore
└─ README.md            # Este archivo (visión general del proyecto)

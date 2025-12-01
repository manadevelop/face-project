# Face Client

Cliente de escritorio para captura de rostro, visualización de malla facial (MediaPipe) y comunicación con el **Face Backend** (FastAPI + DeepFace) para:

- Enrolar personas (registro de embeddings).
- Verificar identidad 1:1.
- Identificar identidad 1:N.

La interfaz está construida con **PyQt5**, la cámara y el procesamiento de imágenes con **OpenCV**, y la detección/malla facial con **MediaPipe Face Detection + Face Mesh**.

---

## 1. Requisitos

- macOS (Apple Silicon, por ejemplo M4 Pro).
- Python 3.12 instalado (por ejemplo vía Homebrew).
- El proyecto `face_backend` debe estar configurado y ejecutándose en:

```text
http://127.0.0.1:8000
```

(si lo cambias, debes actualizar la constante `BACKEND_URL` en `client_gui.py`).

---

## 2. Estructura del proyecto

```text
face_client/
├─ venv/               # Entorno virtual (no se versiona)
├─ client_gui.py       # Código principal del cliente
└─ README.md
```

---

## 3. Configuración del entorno virtual

Desde la carpeta `face_client`:

```bash
cd /ruta/a/face_client

# Crear entorno virtual con Python 3.12
/opt/homebrew/opt/python@3.12/bin/python3.12 -m venv venv

# Activar el entorno virtual (macOS / Linux)
source venv/bin/activate

# Actualizar pip
pip install --upgrade pip
```

---

## 4. Instalación de dependencias

Con el entorno virtual activado:

```bash
pip install pyqt5 opencv-contrib-python mediapipe==0.10.14 "protobuf>=4.25.3,<5" numpy requests
```

**Librerías principales:**

- `pyqt5` → interfaz gráfica de escritorio.
- `opencv-contrib-python` → acceso a cámara y procesamiento de imágenes.
- `mediapipe` (Face Detection + Face Mesh) → detección del rostro y malla de puntos característicos.
- `numpy` → manejo de arreglos numéricos.
- `requests` → envío de imágenes y datos al backend vía HTTP.

---

## 5. Archivo principal: `client_gui.py`

El archivo `client_gui.py` implementa:

- Ventana principal `FaceClient` (hereda de `QMainWindow`).
- Captura de video en vivo desde la cámara (`cv2.VideoCapture(0)`).
- Detección de rostro con `MediaPipe FaceDetection`.
- Malla de puntos faciales con `MediaPipe FaceMesh`.
- Recorte del rostro detectado y envío al backend como imagen JPEG.
- Modos de operación:
  - `ENROLAR`
  - `VERIFICAR 1:1`
  - `IDENTIFICAR 1:N`
- Botón **Detener cámara / Reanudar cámara** que muestra una pantalla de información del proyecto.

La URL del backend se configura mediante la constante:

```python
BACKEND_URL = "http://127.0.0.1:8000"
```

---

## 6. Ejecución

### 6.1 Ejecutar el backend

En una terminal (en la carpeta `face_backend`):

```bash
cd /ruta/a/face_backend
source venv/bin/activate
uvicorn main:app --reload
```

El backend quedará disponible en:

- API base: <http://127.0.0.1:8000>
- Documentación Swagger: <http://127.0.0.1:8000/docs>

### 6.2 Ejecutar el cliente

En otra terminal, dentro de `face_client`:

```bash
cd /ruta/a/face_client
source venv/bin/activate
python client_gui.py
```

Si en tu sistema el comando es `python3`, entonces:

```bash
python3 client_gui.py
```

---

## 7. Modos de operación

En la parte derecha de la interfaz hay un combo **Modo** con tres opciones:

### 7.1 ENROLAR

- Selecciona `ENROLAR`.
- Escribe un **Nombre / ID** (por ejemplo, un DNI o código interno).
- Alinea tu rostro dentro del recuadro verde (se verá la malla facial).
- Pulsa **“Capturar y enviar”**.

El cliente:

1. Recorta el rostro detectado.
2. Lo convierte a JPEG.
3. Envía un `POST /enroll` al backend con:
   - `person_id` = Nombre/ID ingresado.
   - `name` = Nombre/ID ingresado.
   - `image` = archivo JPEG del rostro.

El backend genera el embedding, lo almacena y responde algo como:

```json
{
  "status": "ok",
  "message": "Enrolado 44922647",
  "total_embeddings": 1
}
```

El cliente muestra en la barra de estado:  
`Estado: Enrolado <ID> (total=N)`.

---

### 7.2 VERIFICAR 1:1

Modo de **verificación**: el usuario ingresa el ID que afirma ser, y el sistema comprueba si el rostro corresponde a ese ID.

- Selecciona `VERIFICAR 1:1`.
- Escribe el **Nombre / ID esperado** (el mismo que usaste al enrolar).
- Enfoca tu rostro.
- Pulsa **“Capturar y enviar”**.

El cliente:

1. Envía la imagen al endpoint `POST /identify` del backend.
2. El backend devuelve el mejor match y su similitud.
3. El cliente compara el `person_id`/`name` devuelto con el ID ingresado:

   - Si coinciden → muestra algo como:

     `Estado: VERIFICADO 1:1 como 44922647 (sim=0.87)`

   - Si no coinciden → muestra:

     `Estado: NO corresponde al ID 44922647. Mejor match=OTRO_ID (sim=0.75)`

Este modo implementa la lógica 1:1 **en el cliente**, reutilizando el endpoint de identificación 1:N del backend.

---

### 7.3 IDENTIFICAR 1:N

Modo de **identificación**: se busca a qué persona enrolada se parece más el rostro capturado.

- Selecciona `IDENTIFICAR 1:N`.
- No es necesario escribir Nombre / ID (se puede dejar vacío).
- Enfoca tu rostro.
- Pulsa **“Capturar y enviar”**.

El cliente:

1. Envía la imagen al endpoint `POST /identify` del backend.
2. El backend calcula el embedding, recorre todos los embeddings almacenados y devuelve el mejor match (persona con mayor similitud).
3. El cliente muestra:

`Estado: mejor match = NOMBRE (sim=0.82)`

La lógica del umbral (por ejemplo, aceptar solo si similitud ≥ 0.6 o 0.7) se puede añadir en esta capa si se desea.

---

## 8. Botón “Detener cámara / Reanudar cámara”

El botón **Detener cámara**:

- Detiene la captura de video (`QTimer` y `cv2.VideoCapture`).
- Muestra en el área de video una **pantalla blanca** con:
  - Nombre del proyecto (por ejemplo, *“Proyecto: BioFace Demo”*).
  - Nombres de los integrantes (por defecto incluye “Marco Nina Aguilar” y un texto editable).
  - Descripción breve del sistema:

    > “Sistema de enrolamiento y verificación biométrica facial usando MediaPipe en el cliente y DeepFace en el backend.”

Al pulsar de nuevo el botón (que cambia a **“Reanudar cámara”**):

- Se vuelve a abrir la cámara.
- Se reinicia el `QTimer`.
- Se retoma el flujo normal de video, detección y malla facial.

Esta pantalla es útil para:

- Pausar la demostración.
- Mostrar créditos del proyecto.
- Explicar brevemente el funcionamiento cuando no hay cámara activa.

---

## 9. Notas y posibles mejoras

- Ajustar el umbral de similitud para decidir si un match es aceptable.
- Mostrar en la interfaz la similitud mínima aceptada.
- Agregar un listado de personas enroladas y la posibilidad de limpiar/gestionar la base desde el cliente.
- Integrar un indicador visual cuando **no** se detecta rostro (por ejemplo, texto rojo en la ventana).

---

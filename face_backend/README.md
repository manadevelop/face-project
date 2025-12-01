# Face Backend

Servicio backend de reconocimiento facial basado en **DeepFace**, expuesto como API REST con **FastAPI**.

Este proyecto forma parte de una arquitectura de dos componentes:

- **Face Backend (`face_backend`)** ← este proyecto.  
- **Face Client (`face_client`)**: aplicación de escritorio que captura el rostro (OpenCV + MediaPipe) y lo envía al backend.

El backend **no hace detección de rostros**; asume que las imágenes recibidas ya son recortes del rostro y se enfoca en:

- Generar vectores de características (embeddings) usando **ArcFace**.
- Enrolar personas (guardar embeddings).
- Identificar 1:N a partir de un nuevo rostro.

---

## 1. Requisitos

- macOS (Apple Silicon, por ejemplo M4 Pro).
- Python 3.12 instalado (por ejemplo vía Homebrew):

```bash
brew install python@3.12
```

---

## 2. Estructura del proyecto

```text
face_backend/
├─ venv/                  # Entorno virtual (no se versiona)
├─ models/
│  └─ embeddings.pkl      # Base de embeddings (se genera en tiempo de ejecución)
├─ main.py                # Código principal FastAPI + DeepFace
└─ README.md
```

La carpeta `models/` se crea automáticamente cuando se guarda el primer embedding.

---

## 3. Configuración del entorno virtual

### 3.1 Crear y activar el entorno virtual

Desde la carpeta `face_backend`:

```bash
cd /ruta/a/face_backend

# Crear entorno virtual con Python 3.12
/opt/homebrew/opt/python@3.12/bin/python3.12 -m venv venv

# Activar el entorno virtual (macOS / Linux)
source venv/bin/activate

# Actualizar pip
pip install --upgrade pip
```

### 3.2 Instalar dependencias del backend

```bash
pip install fastapi "uvicorn[standard]" deepface opencv-python-headless numpy
pip install python-multipart tf-keras
```

**Librerías principales:**

- `fastapi` → framework web para construir la API REST.
- `uvicorn` → servidor ASGI para ejecutar FastAPI.
- `deepface` → framework de alto nivel para reconocimiento facial (ArcFace, Facenet, etc.).
- `tensorflow` / `tf-keras` → backend de deep learning utilizado por DeepFace.
- `opencv-python-headless` → procesamiento de imágenes (sin GUI).
- `numpy` → operaciones numéricas (vectores y matrices).
- `python-multipart` → permite recibir archivos e inputs de formulario (`UploadFile`, `Form`) en FastAPI.

---

## 4. Ejecución del backend

Con el entorno virtual activado:

```bash
uvicorn main:app --reload
```

Por defecto, el servicio queda disponible en:

- API base: <http://127.0.0.1:8000>  
- Documentación interactiva (Swagger UI): <http://127.0.0.1:8000/docs>  
- Esquema OpenAPI: <http://127.0.0.1:8000/openapi.json>

---

## 5. Endpoints disponibles

### 5.1 `GET /health`

Verifica el estado del servicio y del modelo de embeddings.

**Request:**

```http
GET /health
```

**Response (200):**

```json
{
  "status": "ok",
  "model": "ArcFace"
}
```

---

### 5.2 `POST /enroll`

Enrola una nueva persona en la base de embeddings.

- **URL:** `POST /enroll`
- **Content-Type:** `multipart/form-data`

**Parámetros:**

- `person_id` (form-data, string): Identificador único de la persona (DNI, código interno, etc.).
- `name` (form-data, string): Nombre para mostrar.
- `image` (file, obligatorio): Imagen del rostro recortado (JPG/PNG), enviada normalmente por el cliente (`face_client`).

**Proceso interno:**

1. Decodifica la imagen con OpenCV.
2. Normaliza tamaño y color (RGB, 224×224).
3. Genera un embedding con DeepFace (modelo `ArcFace`, `detector_backend="skip"`).
4. Guarda el embedding y los datos de la persona en `models/embeddings.pkl`.

**Response (200):**

```json
{
  "status": "ok",
  "message": "Enrolado Marco",
  "total_embeddings": 5
}
```

---

### 5.3 `POST /identify`

Realiza identificación 1:N buscando el mejor match en la base de embeddings.

- **URL:** `POST /identify`
- **Content-Type:** `multipart/form-data`

**Parámetros:**

- `image` (file, obligatorio): Imagen de rostro recortado (JPG/PNG).

**Proceso interno:**

1. Decodifica y normaliza la imagen igual que en `/enroll`.
2. Genera un embedding para la imagen recibida.
3. Carga todos los embeddings existentes desde `models/embeddings.pkl`.
4. Calcula la similaridad coseno entre el embedding de consulta y cada embedding guardado.
5. Devuelve el mejor match y su similitud.

**Response (200):**

```json
{
  "person_id": "Marco",
  "name": "Marco",
  "similarity": 0.87
}
```

La decisión final (match aceptado / rechazado) se puede aplicar en capas superiores usando un umbral de similitud (por ejemplo 0.6, 0.7, etc.).

---

## 6. Implementación interna (resumen)

### 6.1 Almacenamiento de embeddings

- Archivo: `models/embeddings.pkl`
- Tipo: `pickle` con una lista de diccionarios de la forma:

```python
[
  {
    "person_id": "MARCO001",
    "name": "Marco",
    "embedding": np.ndarray([...], dtype=float32)
  },
  ...
]
```

Este almacenamiento es simple y local, ideal para pruebas y desarrollo.  
En un entorno más robusto se recomienda utilizar una base de datos (por ejemplo, PostgreSQL).

### 6.2 Modelo de DeepFace

Actualmente el backend usa:

- Modelo: `ArcFace`
- Configuración: `detector_backend="skip"`

La detección y recorte del rostro se realizan en el cliente (por ejemplo, con MediaPipe), por lo que el backend solo recibe rostros ya recortados.

---

## 7. Flujo típico (junto al Face Client)

1. El **Face Client** abre la cámara, detecta el rostro con MediaPipe y dibuja landmarks.
2. El cliente recorta el rostro detectado y lo codifica como imagen JPEG.
3. Para **enrolar**:
   - Envía `person_id`, `name` e `image` al endpoint `POST /enroll` del backend.
4. Para **identificar**:
   - Envía solo la `image` al endpoint `POST /identify`.
   - El backend devuelve el mejor candidato y la similitud correspondiente.
5. El cliente muestra en su interfaz el nombre de la persona y el valor de similitud, y aplica sus reglas de aceptación/rechazo (por ejemplo, si la similitud es mayor a cierto umbral).

# Skin Analysis App

A small Flask app that analyzes uploaded face photos to predict skin type and provide product suggestions. The app uses MediaPipe for face detection and a Keras/TensorFlow model for skin-type prediction.

**Project Root**: `c:\Users\user\Desktop\skin-analysis-app`

**Main app file**: `app.py`

**Templates**: `templates/` (contains `index.html`, `result.html`, error pages)

**Static assets**: `static/` (CSS, JS, uploads)

## Requirements

- Python 3.8+ recommended
- Packages used (install via pip): `flask`, `opencv-python`, `mediapipe`, `tensorflow`, `numpy`, `pillow`, `h5py`

You can install them directly:

PowerShell

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install flask opencv-python mediapipe tensorflow numpy pillow h5py
```

macOS / Linux

```bash
python3 -m venv venv
source venv/bin/activate
pip install flask opencv-python mediapipe tensorflow numpy pillow h5py
```

Optionally create a `requirements.txt` with the above package names for reproducibility.

## Models

- `detector.tflite` — MediaPipe face detector model. The app will automatically download this file at first run if it is not present.
- `skin_type_model.h5` — Keras HDF5 model used to predict skin type. This file must exist in the project root (same folder as `app.py`).
- `acne_model.h5` — (optional) if your copy of the app expects an acne model, place it in the project root or set a custom path via environment variable (see below).

Where to place your models

- Put your models in the project root, for example:
  - `c:\Users\user\Desktop\skin-analysis-app\skin_type_model.h5`
  - `c:\Users\user\Desktop\skin-analysis-app\acne_model.h5`

Using a custom model path

- You can set `MODEL_PATH` (or another env var used by the app) before running the app. Example (PowerShell):

```powershell
#$env:MODEL_PATH for a custom acne model (if the app checks this env var)
$env:MODEL_PATH = 'C:\path\to\your\acne_model.h5'
python app.py
```

If the app is modified to read `MODEL_PATH` for the acne model it will load that file instead of a fixed filename.

## Running the App

Simplest (recommended):

```powershell
# Activate venv (if created earlier)
.\venv\Scripts\Activate.ps1
python app.py
```

Alternative using `flask run` (PowerShell):

```powershell
$env:FLASK_APP = 'app.py'
$env:FLASK_ENV = 'development'
flask run
```

Note: On Windows PowerShell you must use `$env:VAR = 'value'` (not `export`).

Open the app in your browser at `http://127.0.0.1:5000/`.

## Common Issues & Troubleshooting

- Error: `FileNotFoundError: ... 'acne_model.h5'` or `skin_type_model.h5`

  - Cause: model file missing from project root.
  - Fix: copy or download the `.h5` file to the project root. Example:
    ```powershell
    Copy-Item 'C:\path\to\your\model.h5' .\acne_model.h5
    Test-Path .\acne_model.h5
    ```
  - Alternative: set an environment variable with the model path and update `app.py` to read that variable.

- If you see MediaPipe model download messages, that's normal: `detector.tflite` will be downloaded automatically.

- If you want the app to continue running without a missing `.h5`, modify the top of `app.py` to load models safely (guard with `os.path.exists(...)`) and handle `None` model references at prediction time. Example snippet:

```python
acne_model = None
model_path = os.environ.get('MODEL_PATH', 'acne_model.h5')
if os.path.exists(model_path):
    acne_model = load_model(model_path)
else:
    print('Warning: acne_model.h5 not found — running without that model')

def predict_acne(face_img):
    if acne_model is None:
        return 'Unknown (model missing)'
    # otherwise run prediction
```

## Code & Routes

- `app.py` contains the Flask routes and logic:
  - `/` — index/upload page
  - `/analyze` — POST route that accepts uploaded images, runs quality checks, detects face with MediaPipe, predicts skin type using `skin_type_model.h5`, returns results and suggestions

## Project Structure (important files)

- `app.py` — main Flask application
- `templates/` — HTML templates: `index.html`, `result.html`, error pages
- `static/` — CSS/JS and `static/uploads` for uploaded files
- `skin_type_model.h5` — required Keras model for skin-type prediction
- `acne_model.h5` — optional acne detection model (if referenced in `app.py`)

## Next Steps / Tips

- If you have a TensorFlow SavedModel, you can convert it to `.h5` with:

```python
from tensorflow import keras
model = keras.models.load_model('path/to/saved_model_dir')
model.save('acne_model.h5')
```

- If your model is in PyTorch or ONNX, either export to TensorFlow/Keras or adapt `app.py` to use the appropriate runtime.
- Consider adding a `requirements.txt` (run `pip freeze > requirements.txt` in your venv) for reproducible installs.


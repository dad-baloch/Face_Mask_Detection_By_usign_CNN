# Face Mask Detection

Public-ready packaging for a simple face mask detector. The repo contains a training notebook, a saved Keras/TensorFlow model, and a reference dataset layout with no personal images. Replace the sample folders with your own data before publishing.

## Repository Layout
- `notebook/Face_Mask_Detection.ipynb` — end-to-end training and evaluation notebook.
- `notebook/face_mask_detection_model.h5` — pretrained weights produced by the notebook.
- `DataSet/with_mask/`, `DataSet/without_mask/` — expected dataset layout (empty or sample-only; add your own images).
- `test/` — drop images here for quick inference checks.
- `requirements.txt` — minimal dependency set for training and inference.

## Setup
1. Install Python 3.9–3.11.
2. (Recommended) Create a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. If you use VS Code or Jupyter, ensure the `.venv` kernel is selected before running the notebook.

## Training
1. Populate `DataSet/with_mask/` and `DataSet/without_mask/` with your own images (no personal data included in this repo).
2. Open `notebook/Face_Mask_Detection.ipynb` and run cells in order to train and validate.
3. The trained model is saved to `notebook/face_mask_detection_model.h5`.

## Inference (Pretrained Model)
Run a quick check on a single image using the saved model:
```python
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

model = tf.keras.models.load_model("notebook/face_mask_detection_model.h5")
img = image.load_img("test/sample.jpg", target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0) / 255.0
pred = model.predict(img_array)[0][0]
label = "with_mask" if pred < 0.5 else "without_mask"
print(label, float(pred))
```

## Publishing Notes
- This README and requirements are cleaned of personal details; add a license file before publishing.
- If you introduce large assets (new datasets or bigger models), add them to `.gitignore` or store them via releases/storage and document the download step.
- Consider exporting to TensorFlow Lite or ONNX for deployment once the model meets your accuracy requirements.

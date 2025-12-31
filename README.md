# Face Mask Detection

Lightweight demo project for detecting whether a person is wearing a face mask. Includes a pre-trained Keras/TensorFlow model, a sample notebook for training/evaluation, and a reference dataset folder layout.

## Project Structure
- `notebook/Face_Mask_Detection.ipynb` — end-to-end training and evaluation notebook.
- `notebook/face_mask_detection_model.h5` — saved model produced by the notebook.
- `DataSet/with_mask/`, `DataSet/without_mask/` — example dataset layout (images organized by class).
- `test/` — place to store images you want to run inference on.
- `requirements.txt` — Python dependencies.

## Quickstart
1) Install Python 3.9+.
2) (Recommended) Create a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```
3) Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4) Open the notebook in VS Code or Jupyter and run cells in order:
   - Load data
   - Train/validate
   - Export `face_mask_detection_model.h5`

## Using the Pretrained Model for Inference
Below is a minimal Python snippet to load the saved model and run prediction on a single image:
```python
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

model = tf.keras.models.load_model("notebook/face_mask_detection_model.h5")
img = image.load_img("test/sample.jpg", target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0) / 255.0
pred = model.predict(img_array)[0][0]
label = "with_mask" if pred < 0.5 else "without_mask"
print(label, float(pred))
```

## Training Notes
- Dataset expects two class folders: `with_mask` and `without_mask`.
- Ensure images are reasonably balanced between classes.
- Adjust image size and augmentation parameters inside the notebook to fit your hardware.

## Tips
- If you add new large assets (images, models), update `.gitignore` accordingly.
- For faster experiments, start with a small subset of images to verify the pipeline.
- Consider exporting to TensorFlow Lite or ONNX for deployment once satisfied with accuracy.

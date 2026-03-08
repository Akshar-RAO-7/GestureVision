# GestureVision DL — Deep Learning Hand Gesture Recognition (CNN + CNN-LSTM)

GestureVision DL is a deep learning–based hand gesture recognition system capable of recognizing both **static and dynamic gestures** in real time. The system uses **CNN for static image classification** and a **CNN-LSTM hybrid model for temporal gesture sequences**, enabling robust gesture detection for applications such as **assistive technologies, AR/VR interfaces, and gesture-based control systems**.

---

## Project Overview

This project implements a full deep learning pipeline for gesture recognition:

**Preprocessing**

* Skin-color segmentation using **HSV color space**
* Optional **background subtraction (MOG2)**
* Image resize to **128 × 128**
* Pixel normalization

**Data Augmentation**

* Random rotation
* Horizontal flipping
* Random brightness and contrast adjustments

**Models**

* **CNN Model** – for static hand gesture images
* **CNN-LSTM Model** – for dynamic gesture sequences

**Training Setup**

* Optimizer: **Adam (learning rate = 0.001)**
* Loss Function: **Categorical Cross Entropy**
* Callbacks:

  * EarlyStopping
  * ReduceLROnPlateau
  * ModelCheckpoint

**Real-Time Inference**

* Webcam-based gesture detection
* Static mode or dynamic sliding-window sequence recognition

---

## Tech Stack

* TensorFlow 2.14+
* OpenCV
* NumPy
* Python 3.9+

---

## Project Structure

```
gesturevision_dl/
├─ .vscode/
│  ├─ launch.json
│  └─ settings.json
├─ config/
│  └─ settings.yaml
├─ dataset/
│  ├─ static/CLASS_NAME/*.jpg
│  └─ dynamic/CLASS_NAME/SEQUENCE_ID/frame_0001.jpg
├─ models/
├─ src/
│  ├─ segmentation.py
│  ├─ preprocess.py
│  ├─ cnn_model.py
│  ├─ cnn_lstm_model.py
│  ├─ train_cnn.py
│  ├─ train_cnn_lstm.py
│  ├─ infer_realtime.py
│  ├─ capture_static.py
│  ├─ capture_dynamic.py
│  └─ utils.py
├─ requirements.txt
├─ Makefile
└─ README.md
```

---

## Installation

Create and activate a virtual environment.

### Windows

```
python -m venv .venv
.venv\Scripts\activate
```

### macOS / Linux

```
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```
pip install -r requirements.txt
```

---

## Data Collection (Optional)

### Capture Static Gesture Images

```
python -m src.capture_static --out dataset/static --labels open_palm fist peace ok
```

### Capture Dynamic Gesture Sequences

```
python -m src.capture_dynamic --out dataset/dynamic --labels open_palm fist peace ok --seq-len 16
```

---

## Model Training

### Train CNN (Static Gestures)

```
python -m src.train_cnn --data dataset/static --epochs 50
```

### Train CNN-LSTM (Dynamic Gestures)

```
python -m src.train_cnn_lstm --data dataset/dynamic --seq-len 16 --epochs 50
```

---

## Real-Time Gesture Recognition

### Static Gesture Recognition

```
python -m src.infer_realtime --mode static --img-size 128
```

### Dynamic Gesture Recognition

```
python -m src.infer_realtime --mode dynamic --seq-len 16 --img-size 128
```

Press **q** to exit the webcam window.

---

## Example One-Command Setup

```
python -m venv .venv && . .venv/Scripts/activate 2>nul || source .venv/bin/activate
pip install -r requirements.txt
python -m src.capture_static --out dataset/static --labels open_palm fist peace ok
python -m src.train_cnn --data dataset/static --epochs 20
python -m src.infer_realtime --mode static --img-size 128
```

---

## Configuration

You can modify parameters such as:

* class labels
* image size
* sequence length
* training epochs

inside:

```
config/settings.yaml
```

---

## Model Outputs

Trained models and metadata will be saved in:

```
models/
```

Files include:

* trained model weights
* label_map.json
* training checkpoints

---

## Applications

GestureVision DL can be applied in:

* Assistive technology for people with disabilities
* AR/VR interaction systems
* Touchless human–computer interaction
* Smart home gesture control
* Robotics control interfaces

---

## License

This project is released under the **MIT License**.

---

## Author

Akshar Rao
B.E. Computer Science (AI/ML)
Chandigarh University

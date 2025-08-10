# License — License Plate Detection (YOLOv8, YOLOv10, YOLOv12)

License plate detection using Ultralytics YOLO models (YOLOv8, YOLOv10, YOLOv12).  
This repository contains the dataset in YOLO format and the commands/scripts used to train, evaluate, and run inference.

---

## 📂 Project Structure

license/
├── data.yaml
├── images/
│ ├── train/
│ └── val/
├── labels/
│ ├── train/
│ └── val/
├── examples/
│ ├── input1.jpg
│ ├── output1.jpg
│ ├── input2.jpg
│ ├── output2.jpg
│ ├── input3.jpg
│ ├── output3.jpg
|__readme.md

yaml
Copy code

---

## ⚙️ Requirements / Installation

# License Plate Anonymization using YOLO Models

---

## Installation

```bash
pip install --upgrade pip
pip install ultralytics
# Optional: GPU-enabled PyTorch (example for CUDA 11.8)
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```
## Check YOLO Version
```
yolo version
```
🗂 Prepare Dataset (YOLO Format)
Example data.yaml file:

```yaml
path: ./license
train: images/train
val: images/val
nc: 1
names: ['license_plate']
```
🚀 Training

YOLOv8
```powershell
yolo task=detect mode=train model=yolov8n.pt data=.\data.yaml epochs=100 imgsz=640 project=.\runs name=results_v8
```

YOLOv10
```powershell
yolo task=detect mode=train model=yolov10n.pt data=.\data.yaml epochs=100 imgsz=640 project=.\runs name=results_v10
```

YOLOv12
```powershell
yolo task=detect mode=train model=yolov12n.pt data=.\data.yaml epochs=100 imgsz=640 project=.\runs name=results_v12
```
📁 Output Location
Each run saves to:

```text
.\runs\detect\<name>\
```

Inside this folder, you'll find:
```text
weights/best.pt          # Best model
weights/last.pt          # Final model
results.png              # Metrics plot
confusion_matrix.png     # Confusion matrix
```

##📊 Model Accuracy & Speed Comparison

| Model    | mAP@0.5 | mAP@0.5:0.95 | Training Time (100 epochs)     | Inference Speed (ms/img) |
|----------|---------|--------------|-------------------------------|--------------------------|
| YOLOv8n  | 0.933   | —            | ~9 hrs (CPU) / ~45 min (GPU)  | —                        |
| YOLOv10n | 0.918   | 0.570        | ~10 hrs (CPU) / ~50 min (GPU) | —                        |
| YOLOv12n | 0.9277  | 0.5836       | ~12 hrs (CPU) / ~55 min (GPU) | 10.22                    |


##📊 Model Metrics
## YOLOv8 Evaluation Metrics

- TP: 134 | FN: 14 | FP: 11 | TN: 0
- TP Rate (Recall): 0.92 | FN Rate: 0.08 | FP Rate: 1.00 | TN Rate: 0.00
- Precision (number_plate): 0.93
- Precision (all classes): 1.00 @ conf = 0.924
- Recall (all classes): 0.94 @ conf = 0.000
- F1-Score (all classes): 0.91 @ conf = 0.342
- mAP@0.5 (number_plate): 0.933
- mAP@0.5 (all classes): 0.933


## YOLOv10 Evaluation Metrics

| Metric        | Value | Meaning                                         |
|---------------|-------|------------------------------------------------|
| Precision (P) | 0.942 | 94.2% detected objects are correct (low FP).  |
| Recall (R)    | 0.791 | 79.1% actual objects detected (some FN).       |
| mAP@0.5       | 0.918 | High accuracy at IoU 0.5                        |
| mAP@0.5:0.95  | 0.570 | Good generalization across stricter IoU levels|

## YOLOv12 Evaluation Metrics

- Precision: 0.8841
- Recall: 0.8946
- mAP@0.5: 0.9277
- mAP@0.5:0.95: 0.5836
- Inference Time: 10.22 ms/image

### Speed Breakdown:

- Preprocess: 0.29 ms/image
- Inference: 10.22 ms/image
- Loss: 0.00 ms/image
- Postprocess: 3.71 ms/image


## 🖼 Results & Sample Predictions

| Model   | Input                 | Output                   |
|---------|-----------------------|--------------------------|
| YOLOv8  | `examples/input1.jpg`  | `examples/output1.jpg`   |
| YOLOv10 | `examples/input2.jpg`  | `examples/output2.jpg`   |
| YOLOv12 | `examples/input3.jpg`  | `examples/output3.jpg`   |


🔍## Inference 
Example for YOLOv8

```powershell
yolo task=detect mode=predict model=.\runs\detect\results_v8\weights\best.pt source=.\examples\input1.jpg save=True
```
For YOLOv10 and YOLOv12, replace results_v8 with results_v10 or results_v12 respectively.

## Export Model (ONNX Format)

### YOLOv8

```powershell
yolo export model=.\runs\detect\results_v8\weights\best.pt format=onnx
```

### YOLOv10

```powershell
yolo export model=.\runs\detect\results_v10\weights\best.pt format=onnx
```

### YOLOv12

```powershell
yolo export model=.\runs\detect\results_v12\weights\best.pt format=onnx
```

## 📊 Evaluation

```powershell
yolo task=detect mode=val model=.\runs\detect\results_v8\weights\best.pt data=.\data.yaml save_json=True
```

## 📜 License & Authors

This project is licensed under the **MIT License**.

### Authors:
- Bibikhuteja Soudagar
- R Prerana
- Megha T
- Sonali Chandake

### Contact:
- bibikhutejasoudagar21@gmail.com
- rprerana777@gmail.com
- meghatalawar22@gmail.com
- sonalichendake21@gmail.com

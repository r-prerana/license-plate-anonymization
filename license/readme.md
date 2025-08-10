# license — License Plate Detection (YOLOv8, YOLOv10, YOLOv12)

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

```powershell
pip install --upgrade pip
pip install ultralytics
# Optional: GPU-enabled PyTorch (example for CUDA 11.8)
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

yolo version
🗂 Prepare Dataset (YOLO Format)
data.yaml example:

yaml
Copy code
path: ./license
train: images/train
val: images/val
nc: 1
names: ['license_plate']
🚀 Training
YOLOv8

powershell
Copy code
yolo task=detect mode=train model=yolov8n.pt data=.\data.yaml epochs=100 imgsz=640 project=.\runs name=results_v8
YOLOv10

powershell
Copy code
yolo task=detect mode=train model=yolov10n.pt data=.\data.yaml epochs=100 imgsz=640 project=.\runs name=results_v10
YOLOv12

powershell
Copy code
yolo task=detect mode=train model=yolov12n.pt data=.\data.yaml epochs=100 imgsz=640 project=.\runs name=results_v12
📁 Output Location
Each run saves to:

php-template
Copy code
.\runs\detect\<name>\
Inside:

weights/best.pt — best model

weights/last.pt — final model

results.png — metrics plot

confusion_matrix.png — confusion matrix

📊 Accuracy & Speed Comparison
Model	mAP@0.5	mAP@0.5:0.95	Training Time (100 epochs)	Inference Speed (ms/img)
YOLOv8n	0.933	—	~9 hrs (CPU) / ~45 min (GPU)	—
YOLOv10n	0.918	0.570	~10 hrs (CPU) / ~50 min (GPU)	—
YOLOv12n	0.9277	0.5836	~12 hrs (CPU) / ~55 min (GPU)	10.22 ms

📊 Model Metrics
YOLOv8 Evaluation Metrics
TP: 134 | FN: 14 | FP: 11 | TN: 0

TP Rate (Recall): 0.92 | FN Rate: 0.08 | FP Rate: 1.00 | TN Rate: 0.00

Precision (number_plate): 0.93

Precision (all classes): 1.00 @ conf = 0.924

Recall (all classes): 0.94 @ conf = 0.000

F1-Score (all classes): 0.91 @ conf = 0.342

mAP@0.5 (number_plate): 0.933

mAP@0.5 (all classes): 0.933

YOLOv10 Evaluation Metrics
Metric	Value	What it Means
Precision (P)	0.942	94.2% of detected objects are correct (low false positives).
Recall (R)	0.791	79.1% of actual objects detected (some false negatives).
mAP@0.5	0.918	High accuracy at IoU 0.5 – excellent!
mAP@0.5:0.95	0.570	Good generalization across stricter IoU thresholds.

YOLOv12 Evaluation Metrics
Precision: 0.8841

Recall: 0.8946

mAP@0.5: 0.9277

mAP@0.5:0.95: 0.5836

Inference Time: 10.22 ms/image

Speed Breakdown:

Preprocess: 0.29 ms/image

Inference: 10.22 ms/image

Loss: 0.00 ms/image

Postprocess: 3.71 ms/image

🖼 Results & Sample Predictions
YOLOv8
Input: examples/input1.jpg
Output: examples/output1.jpg

YOLOv10
Input: examples/input2.jpg
Output: examples/output2.jpg

YOLOv12
Input: examples/input3.jpg
Output: examples/output3.jpg

🔍 Inference
Example for YOLOv8:

powershell
Copy code
yolo task=detect mode=predict model=.\runs\detect\results_v8\weights\best.pt source=.\examples\input1.jpg save=True
(Same for YOLOv10 and YOLOv12, just change results_v8 to results_v10 or results_v12.)

📤 Export Model (ONNX)
powershell
Copy code
# YOLOv8
yolo export model=.\runs\detect\results_v8\weights\best.pt format=onnx

# YOLOv10
yolo export model=.\runs\detect\results_v10\weights\best.pt format=onnx

# YOLOv12
yolo export model=.\runs\detect\results_v12\weights\best.pt format=onnx
📊 Evaluation
powershell
Copy code
yolo task=detect mode=val model=.\runs\detect\results_v8\weights\best.pt data=.\data.yaml save_json=True
📜 License & Authors

MIT License

Authors:
- Bibikhuteja Soudagar
- R Prerana
- Megha T
- Sonali Chandake

Contact:
- bibikhutejasoudagar21@gmail.com
- rprerana777@gmail.com
- meghatalawar22@gmail.com
- sonalichendake21@gmail.com
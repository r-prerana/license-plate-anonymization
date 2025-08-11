import os
import cv2
from ultralytics import YOLO  # Works with YOLOv10 models

def blur_detections_yolov10(
    model_path,
    input_path,
    output_path,
    blur_kernel=(51, 51),
    device="cpu",
    conf_threshold=0.25
):
    model = YOLO(model_path)

    def process_frame(frame):
        results = model(frame, device=device, conf=conf_threshold)
        for res in results:
            for box in res.boxes.xyxy:
                x1, y1, x2, y2 = map(int, box[:4])
                roi = frame[y1:y2, x1:x2]
                if roi.size > 0:
                    roi = cv2.GaussianBlur(roi, blur_kernel, 0)
                    frame[y1:y2, x1:x2] = roi
        return frame

    if os.path.isdir(input_path):
        os.makedirs(output_path, exist_ok=True)
        for fname in os.listdir(input_path):
            if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                img = cv2.imread(os.path.join(input_path, fname))
                if img is not None:
                    cv2.imwrite(
                        os.path.join(output_path, fname),
                        process_frame(img)
                    )
                    print(f"Saved: {fname}")
    else:
        ext = os.path.splitext(input_path)[1].lower()
        if ext in ('.mp4', '.avi', '.mov'):
            cap = cv2.VideoCapture(input_path)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(
                output_path, fourcc,
                cap.get(cv2.CAP_PROP_FPS),
                (int(cap.get(3)), int(cap.get(4)))
            )
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                out.write(process_frame(frame))
            cap.release()
            out.release()
            print(f"Video saved: {output_path}")
        else:
            img = cv2.imread(input_path)
            if img is not None:
                cv2.imwrite(output_path, process_frame(img))
                print(f"Output saved: {output_path}")

    print("Done.")

# Example Usage
MODEL_PATH = "path/to/yolov10n.pt"
blur_detections_yolov10(
    model_path=MODEL_PATH,
    input_path="path/to/input_folder_or_file",
    output_path="path/to/output",
    device="cpu",
    blur_kernel=(51, 51),
    conf_threshold=0.25
)
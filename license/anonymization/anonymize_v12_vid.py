import os
import cv2
from ultralytics import YOLO

def blur_with_yolov12(
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
                    processed = process_frame(img)
                    cv2.imwrite(os.path.join(output_path, fname), processed)
                    print(f"Saved: {fname}")

    elif input_path.lower().endswith(('.mp4', '.avi', '.mov')):
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {input_path}")
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
        print("Processing video...")
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            out.write(process_frame(frame))
        cap.release()
        out.release()
        print(f"Video saved: {output_path}")

    else:
        img = cv2.imread(input_path)
        if img is None:
            raise FileNotFoundError(f"Cannot open image: {input_path}")
        processed = process_frame(img)
        cv2.imwrite(output_path, processed)
        print(f"Image saved: {output_path}")

    print("✅ Done.")

#  Example usage:
MODEL_PATH = "path/to/yolov12n.pt" 
blur_with_yolov12(
    model_path=MODEL_PATH,
    input_path="path/to/input_or_folder",
    output_path="path/to/output_or_folder",
    device=0,  # GPU index or "cpu"
    blur_kernel=(51, 51),
    conf_threshold=0.25
)
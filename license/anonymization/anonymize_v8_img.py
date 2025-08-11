import cv2
import os
from ultralytics import YOLO

def blur_detections_yolov8(
        model_path,
        input_path,
        output_path,
        blur_kernel=(51, 51),
        device="cpu",
        conf_threshold=0.25
    ):
    """
    Generalized YOLOv8 detection & blur script.

    model_path      : Path to YOLOv8 model (.pt file)
    input_path      : Path to image, video, or folder
    output_path     : Output file/folder path
    blur_kernel     : Tuple for Gaussian blur strength (odd numbers)
    device          : 'cpu' or GPU index (0, 1, ...)
    conf_threshold  : Confidence threshold for detection
    """

    # Load YOLOv8 model
    model = YOLO(model_path)

    def process_frame(frame):
        results = model(frame, device=device, conf=conf_threshold)
        for result in results:
            for box in result.boxes.xyxy:
                x1, y1, x2, y2 = map(int, box[:4])
                roi = frame[y1:y2, x1:x2]
                if roi.size > 0:
                    roi = cv2.GaussianBlur(roi, blur_kernel, 0)
                    frame[y1:y2, x1:x2] = roi
        return frame

    # Check if input is file or folder
    if os.path.isdir(input_path):  
        os.makedirs(output_path, exist_ok=True)
        for filename in os.listdir(input_path):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                img = cv2.imread(os.path.join(input_path, filename))
                if img is not None:
                    processed = process_frame(img)
                    cv2.imwrite(os.path.join(output_path, filename), processed)
                    print(f"Saved: {filename}")
    else:
        if input_path.lower().endswith(('.mp4', '.avi', '.mov')):
            cap = cv2.VideoCapture(input_path)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc,
                                  cap.get(cv2.CAP_PROP_FPS),
                                  (int(cap.get(3)), int(cap.get(4))))
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                processed = process_frame(frame)
                out.write(processed)
            cap.release()
            out.release()
            print(f"Video saved at {output_path}")
        else:
            img = cv2.imread(input_path)
            if img is not None:
                processed = process_frame(img)
                cv2.imwrite(output_path, processed)
                print(f"Image saved at {output_path}")

    print("✅ Done")


# -------- Example Usage --------
MODEL_PATH = r"C:\Users\bibik\OneDrive\Desktop\test1\best.pt"

# Example 1: Process all images in folder
blur_detections_yolov8(
    model_path=MODEL_PATH,
    input_path=r"C:\Users\bibik\OneDrive\Desktop\test1",
    output_path=r"C:\Users\bibik\OneDrive\Desktop\test1\blurred",
    device=0
)

# Example 2: Process single image
# blur_detections_yolov8(MODEL_PATH, "input.jpg", "output.jpg", device=0)

# Example 3: Process video
# blur_detections_yolov8(MODEL_PATH, "input.mp4", "output.mp4", device=0)

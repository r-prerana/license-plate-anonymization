import cv2
import os
from ultralytics import YOLO

def blur_with_yolov8(model_path, input_path, output_path,
                     blur_kernel=(51, 51), device="cpu", conf_threshold=0.25):
    """
    Generalized YOLOv8 blurring script.
    - Works with images, videos, or folders of images
    - Blurs detected objects based on YOLOv8 bounding boxes
    """

    # Load YOLOv8 model
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

    # Case 1: Input is a folder of images
    if os.path.isdir(input_path):
        os.makedirs(output_path, exist_ok=True)
        for fname in os.listdir(input_path):
            if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                img = cv2.imread(os.path.join(input_path, fname))
                if img is not None:
                    processed = process_frame(img)
                    cv2.imwrite(os.path.join(output_path, fname), processed)
                    print(f"Saved: {fname}")

    # Case 2: Input is a video file
    elif input_path.lower().endswith(('.mp4', '.avi', '.mov')):
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video file: {input_path}")

        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

        print("Processing video...")
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            processed = process_frame(frame)
            out.write(processed)

        cap.release()
        out.release()
        print(f"Video saved: {output_path}")

    # Case 3: Input is a single image
    else:
        img = cv2.imread(input_path)
        if img is None:
            raise FileNotFoundError(f"Cannot open image file: {input_path}")
        processed = process_frame(img)
        cv2.imwrite(output_path, processed)
        print(f"Image saved: {output_path}")

    print("✅ Done.")


# -------- Example usage --------
MODEL_PATH = r"C:\Users\bibik\OneDrive\Desktop\test\best.pt"

# For video
blur_with_yolov8(
    model_path=MODEL_PATH,
    input_path=r"C:\Users\bibik\OneDrive\Desktop\test\input.mp4",
    output_path=r"C:\Users\bibik\OneDrive\Desktop\test\output.mp4",
    device=0
)

# For folder of images
# blur_with_yolov8(MODEL_PATH, "path/to/images_folder", "path/to/output_folder", device=0)

# For single image
# blur_with_yolov8(MODEL_PATH, "input.jpg", "output.jpg", device=0)
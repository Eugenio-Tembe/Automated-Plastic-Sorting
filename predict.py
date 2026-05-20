import cv2
from ultralytics import YOLO

def detect_plastic(image_path, model_path='yolov8n.pt'):
    """
    Loads a trained YOLO model and runs object detection on a plastic sample image.
    Prints out detected classes and confidence scores.
    """
    print(f"[INFO] Loading model from {model_path}...")
    # Load the YOLOv8 model (defaults to nano pre-trained on COCO if no local weights specified)
    model = YOLO(model_path)
    
    print(f"[INFO] Running inference on {image_path}...")
    # Run inference on the image
    results = model(image_path)
    
    # Process and display results
    for result in results:
        boxes = result.boxes
        for box in boxes:
            # Get class ID and map it to name
            class_id = int(box.cls[0])
            class_name = model.names[class_id]
            confidence = float(box.conf[0])
            
            print(f"[DETECTED] Object: {class_name} | Confidence: {confidence:.2f}")
            
    print("[INFO] Inference complete.")

if __name__ == "__main__":
    # Placeholder paths for testing - update these with your local file paths
    TEST_IMAGE = "sample_plastic.jpg" 
    MODEL_WEIGHTS = "best.pt"  # This will be your trained weights from Roboflow/Colab
    
    # Note: To run this locally, ensure you have run: pip install ultralytics opencv-python
    print("--- Maputo Automated Plastic Sorting Assistant ---")
    print("System initialized. Ready for local deployment testing.")

import cv2
import numpy as np
import onnxruntime as ort

# ---------------- CONFIG ---------------- #
MODEL_PATH = "models2/best.onnx"
IMAGE_PATH = "data/images/test3.jpg"

INPUT_SIZE = 640
CONF_THRESHOLD = 0.2   
NMS_THRESHOLD = 0.5
# ---------------------------------------- #

def preprocess(image):
    img = cv2.resize(image, (INPUT_SIZE, INPUT_SIZE))
    img = img / 255.0
    img = img.transpose(2, 0, 1)
    img = np.expand_dims(img, axis=0).astype(np.float32)
    return img

def postprocess(output, img_shape):
    predictions = np.squeeze(output[0]).T  # (8400, 5)

    boxes = []
    scores = []

    h, w = img_shape

    for pred in predictions:
        obj_conf = pred[4]

        if obj_conf < CONF_THRESHOLD:
            continue

        confidence = obj_conf

        cx, cy, bw, bh = pred[0:4]

        
        if cx <= 1.0 and cy <= 1.0:
            # normalized (0–1)
            x = int((cx - bw / 2) * w)
            y = int((cy - bh / 2) * h)
            bw = int(bw * w)
            bh = int(bh * h)
        else:
            # pixel scale (0–640)
            x = int((cx - bw / 2) * w / INPUT_SIZE)
            y = int((cy - bh / 2) * h / INPUT_SIZE)
            bw = int(bw * w / INPUT_SIZE)
            bh = int(bh * h / INPUT_SIZE)

    
        if bw < 5 or bh < 5:
            continue

        boxes.append([x, y, bw, bh])
        scores.append(float(confidence))

    print(f"Total boxes before NMS: {len(boxes)}")

    
    indices = cv2.dnn.NMSBoxes(boxes, scores, CONF_THRESHOLD, NMS_THRESHOLD)

   
    if indices is None or len(indices) == 0:
        print("⚠️ NMS removed all boxes, showing raw detections")
        indices = list(range(len(boxes)))

    return boxes, scores, indices

def draw_detections(image, boxes, scores, indices):
    if indices is None or len(indices) == 0:
        print("❌ No elephant detected")
        return image

    for idx in np.array(indices).flatten():
        x, y, w, h = boxes[idx]

        label = f"Elephant: {scores[idx]:.2f}"

        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(image, label, (x, max(y - 10, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    print("✅ Elephant detected with boxes!")
    return image

def resize_for_display(image, max_width=900):
    h, w = image.shape[:2]
    scale = max_width / w
    return cv2.resize(image, (int(w * scale), int(h * scale)))

def main():
    # Load ONNX model
    session = ort.InferenceSession(MODEL_PATH)
    input_name = session.get_inputs()[0].name

    # Load image
    image = cv2.imread(IMAGE_PATH)
    if image is None:
        print("❌ Image not found")
        return

    orig = image.copy()
    h, w = image.shape[:2]

    # Preprocess
    input_tensor = preprocess(image)

    # Inference
    outputs = session.run(None, {input_name: input_tensor})

    print("Output shape:", outputs[0].shape)

    # Postprocess
    boxes, scores, indices = postprocess(outputs, (h, w))

    # Draw detections
    result = draw_detections(orig, boxes, scores, indices)

    # Show result
    cv2.namedWindow("Elephant Detection", cv2.WINDOW_NORMAL)
    display_img = resize_for_display(result)
    cv2.imshow("Elephant Detection", display_img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Save output
    cv2.imwrite("outputs/images/result.jpg", result)
    print("📁 Result saved to outputs/images/result.jpg")

if __name__ == "__main__":
    main()
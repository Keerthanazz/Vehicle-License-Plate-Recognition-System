from flask import Flask, request, jsonify
import cv2
import numpy as np
import pytesseract
from PIL import Image
import io

app = Flask(__name__)

# Load YOLO
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

def detect_license_plate(image):
    height, width, channels = image.shape
    blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)
    
    class_ids = []
    confidences = []
    boxes = []
    
    for out in outs:
        for detection in out:
            for obj in detection:
                scores = obj[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                
                if confidence > 0.5:
                    center_x = int(obj[0] * width)
                    center_y = int(obj[1] * height)
                    w = int(obj[2] * width)
                    h = int(obj[3] * height)
                    
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
                    
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    
    for i in indices:
        box = boxes[i[0]]
        x, y, w, h = box
        roi = image[y:y+h, x:x+w]
        return roi
    return None

def recognize_text(image):
    text = pytesseract.image_to_string(image, config='--psm 8')
    return text

@app.route('/upload', methods=['POST'])
def upload_file():
    file = request.files['file']
    img = Image.open(io.BytesIO(file.read()))
    img_np = np.array(img)
    plate_image = detect_license_plate(img_np)
    
    if plate_image is not None:
        text = recognize_text(plate_image)
        return jsonify({"text": text})
    else:
        return jsonify({"text": "No license plate detected."})

if __name__ == '__main__':
    app.run(debug=True)

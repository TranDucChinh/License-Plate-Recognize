from ultralytics import YOLO
import cv2
from PIL import Image
import easyocr

def postprocess(text):
    return ''.join(c for c in text.upper() if c.isalnum() or c == '-')
    return text

# Load the YOLO model for license plate detection
license_plate_detector = YOLO('license_plate_detector.pt')

# Load the EasyOCR model for text recognition
reader = easyocr.Reader(['en'], gpu=False)

img_path = 'Image_test/car3.png'
img = cv2.imread(img_path)

results = license_plate_detector.predict(source=img_path)
for result in results:
    boxes = result.boxes.xyxy.cpu().numpy().astype(int)
    for box in boxes:
        x1, y1, x2, y2 = box[:4]
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    for box in boxes:
        x1, y1, x2, y2 = box[:4]
        cropped = img[y1:y2, x1:x2]
        img_gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
        result = reader.readtext(img_gray)
        for bbox, text, conf in result:
            text = postprocess(text)
            cv2.putText(img, text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).show()
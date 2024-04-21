import io
import cv2
import json
import requests
from flask import Flask, render_template, Response, jsonify

app = Flask(__name__, template_folder='template')
plate_number = ""

def detect_plates():
    harcascade = "model/haarcascade_russian_plate_number.xml"
    cap = cv2.VideoCapture(0)
    cap.set(3, 320) # width
    cap.set(4, 240) # height
    min_area = 500

    while True:
        success, img = cap.read()

        plate_cascade = cv2.CascadeClassifier(harcascade)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        plates = plate_cascade.detectMultiScale(img_gray, 1.1, 4)

        
    for (x, y, w, h) in plates:
        area = w * h

        if area > min_area:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(img, "Number Plate", (x, y - 5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 255), 2)

            img_roi = img[y: y + h, x:x + w]

            # Send image to OCR API if plate is detected
            ocr_result = perform_ocr(img_roi)

            # Check if 'plate_number' key exists in the OCR result
            if 'plate_number' in ocr_result:
                global plate_number
                plate_number = ocr_result['plate_number']
            else:
                plate_number = "Plate number not detected"

        ret, jpeg = cv2.imencode('.tiff', img)
        frame = jpeg.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def perform_ocr(img_roi):
    headers = {
        "Authorization": "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VyX2lkIjoiOTVjYzIzZWQtNTgyMS00MDBjLTg4YjQtOGY1YzNlYWQzMWI4IiwidHlwZSI6ImFwaV90b2tlbiJ9.61kGebuxg0wwh0X9amoddfT5DOiSbowTgv7FKCzHDuM",
        "Accept": "application/json"
    }
    url = "https://api.edenai.run/v2/ocr/ocr_async"
    data = {"providers": "amazon"}
    
    # Convert the NumPy array to bytes
    _, img_encoded = cv2.imencode('.tiff', img_roi)
    img_bytes = img_encoded.tobytes()

    files = {'file': ('plate.jpg', img_bytes, 'image/tiff')}
    
    response = requests.post(url, data=data, files=files, headers=headers)
    result = json.loads(response.text)
    
    return result

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(detect_plates(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/ocr_result')
def ocr_result():
    global plate_number
    return jsonify({'plate_number': plate_number})

if __name__ == "__main__":
    app.run(debug=True)

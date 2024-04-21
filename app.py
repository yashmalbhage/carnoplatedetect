from flask import Flask, render_template, Response, jsonify
import cv2
import pytesseract

app = Flask(__name__, template_folder='template')

# Path to the trained cascade classifier
harcascade = "model/haarcascade_russian_plate_number.xml"

# Initialize video capture
cap = cv2.VideoCapture(0)

# Set the width and height of the video capture
cap.set(3, 320)  # width
cap.set(4, 2400)  # height

min_area = 500
count = 0

# Global variable to store the detected plate information
detected_plate = "Waiting for detection..."
vehicle_number = "Waiting for detection..."  # Added for displaying the vehicle number
def detect_plate(skip_frames=10):  # Process every 10th frame
    global detected_plate, vehicle_number
    frame_count = 0
    while True:
        success, img = cap.read()
        if not success:
            break
        
        frame_count += 1
        if frame_count % skip_frames != 0:
            continue

        plate_cascade = cv2.CascadeClassifier(harcascade)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        plates = plate_cascade.detectMultiScale(img_gray, 1.1, 4)

        for (x, y, w, h) in plates:
            area = w * h

            if area > min_area:
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(img, "Number Plate", (x, y - 5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 255), 2)

                img_roi = img[y: y + h, x:x + w]
                
                # Perform OCR to extract vehicle number
                vehicle_number = pytesseract.image_to_string(img_roi, config='--psm 6')

                # Update detected plate information
                detected_plate = "ABC123"  # Replace this with your actual plate detection logic

        # Encode the frame to JPEG
        ret, jpeg = cv2.imencode('.jpg', img)
        frame = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(detect_plate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/plate_info')
def plate_info():
    return jsonify({'detected_plate': detected_plate, 'vehicle_number': vehicle_number})


if __name__ == '__main__':
    app.run(debug=True)

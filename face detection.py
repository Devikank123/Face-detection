import cv2

# Load Haar cascade for face detection
haar_cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
hcas = cv2.CascadeClassifier(haar_cascade_path)

# Initialize webcam
cam = cv2.VideoCapture(0)

while True:
    ret, img = cam.read()
    if not ret:
        print("Failed to capture image")
        break

    # Convert to grayscale
    gimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = hcas.detectMultiScale(gimg, scaleFactor=1.3, minNeighbors=4)

    # Draw rectangles around detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # Show output
    cv2.imshow("Face Detection", img)

    # Exit when ESC key is pressed
    key = cv2.waitKey(10)
    if key == 27:  # ESC key
        break

# Release camera and close windows
cam.release()
cv2.destroyAllWindows()

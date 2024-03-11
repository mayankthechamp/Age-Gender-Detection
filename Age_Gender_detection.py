import cv2

# Load the pre-trained age and gender classification models from OpenCV
age_model = cv2.dnn.readNetFromCaffe(
    '/home/mayank/PycharmProjects/melomaniac2oo3/deploy_age.prototxt',
    '/home/mayank/PycharmProjects/melomaniac2oo3/age_net.caffemodel'
)
gender_model = cv2.dnn.readNetFromCaffe(
    '/home/mayank/PycharmProjects/melomaniac2oo3/gender_deploy.prototxt',
    '/home/mayank/PycharmProjects/melomaniac2oo3/gender_net.caffemodel'
)

# List of age and gender labels
age_labels = ['(0-2)', '(4-6)', '(8-12)', '(15-19)', '(20-25)', '(38-43)', '(48-53)', '(60-100)']
gender_labels = ['Male', 'Female']

# OpenCV initialization
cap = cv2.VideoCapture(0)
while True:
    # Read frame from webcam
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the image to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Perform face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # If faces are detected
    for (x, y, w, h) in faces:
        # Extract the face region of interest
        face_roi = frame[y:y + h, x:x + w]

        # Preprocess the face ROI for age and gender classification
        face_blob = cv2.dnn.blobFromImage(face_roi, 1.0, (227, 227), (78.4263377603, 87.7689143744, 114.895847746),
                                          swapRB=False)

        # Set the input to the age and gender classification models
        age_model.setInput(face_blob)
        gender_model.setInput(face_blob)

        # Perform forward pass and get the predictions
        age_predictions = age_model.forward()
        gender_predictions = gender_model.forward()

        # Get the predicted age and gender labels
        age_index = age_predictions[0].argmax()
        gender_index = gender_predictions[0].argmax()

        age = age_labels[age_index]
        gender = gender_labels[gender_index]

        # Display the age and gender information on the frame
        cv2.putText(frame, f"Age: {age}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Gender: {gender}", (x, y + h + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Draw the face bounding box on the frame
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Display the resulting frame
    cv2.imshow('Age and Gender Detection', frame)

    # Exit loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()

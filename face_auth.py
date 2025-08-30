import cv2
import face_recognition
import numpy as np

# --- Load known face and name ---
KNOWN_IMAGE_PATH = r"C:\Users\91778\OneDrive\Desktop\project\face authentication\ganesh (2).jpg"  # Update this path as needed
KNOWN_NAME = "Ganesh"  # Change the name to match the person

# Load the image and encode the face
known_image = face_recognition.load_image_file(KNOWN_IMAGE_PATH)
known_face_encodings = face_recognition.face_encodings(known_image)
if not known_face_encodings:
    print("No face found in the known image!")
    exit()
known_face_encoding = known_face_encodings[0]

# --- Start webcam ---
video_capture = cv2.VideoCapture(0)
if not video_capture.isOpened():
    print("Could not open webcam. Check your camera.")
    exit()

while True:
    ret, frame = video_capture.read()
    if not ret:
        print("Failed to grab frame from webcam.")
        break

    # Resize frame to 1/4 size for faster processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = small_frame[:, :, ::-1]

    # Find all faces in the frame
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    # Loop over each face found in the frame
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # Compare the detected face with the known face encoding
        match = face_recognition.compare_faces([known_face_encoding], face_encoding)[0]
        name = KNOWN_NAME if match else "Unknown"

        # Scale the face boxes back to original size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw rectangle and name
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 10, bottom - 10), font, 0.75, (255, 255, 255), 1)

    # Show the result
    cv2.imshow('Face Authentication', frame)

    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
video_capture.release()
cv2.destroyAllWindows()


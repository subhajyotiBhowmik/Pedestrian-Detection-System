import cv2
import serial
import pygame

# Load pedestrian detection cascade classifier
pedestrian_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_fullbody.xml')

# Establish serial communication with Arduino
#arduino = serial.Serial('/dev/cu.usbmodem101', 9600)  # Change 'COM3' to the appropriate port

# Initialize Pygame
pygame.init()

# Load the audio file (WAV format)
audio_file = "Kulpi 75.wav"
pygame.mixer.music.load(audio_file)

# Function to play the audio
def play_audio():
    pygame.mixer.music.play()

# Function to detect pedestrians in an image
def detect_pedestrians(image):
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect pedestrians
    pedestrians = pedestrian_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Draw rectangles around detected pedestrians
    for (x, y, w, h) in pedestrians:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display message indicating pedestrians are detected
    if len(pedestrians) > 0:
        cv2.putText(image, "Pedestrians Detected: " + str(len(pedestrians)), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    return image, len(pedestrians)

# Capture video from webcam
cap = cv2.VideoCapture(0)

while True:
    # Read frame from webcam
    ret, frame = cap.read()

    # Check if frame is captured successfully
    if not ret:
        break

    # Detect pedestrians in the frame
    frame_with_pedestrians, num_pedestrians = detect_pedestrians(frame)

    # Display the frame with pedestrian detection
    cv2.imshow('Pedestrian Detection', frame_with_pedestrians)

    # Send pedestrian count to Arduino
    #arduino.write(str(num_pedestrians).encode())

    # Exit loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
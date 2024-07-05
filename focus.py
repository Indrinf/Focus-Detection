import cv2
import numpy as np
from collections import deque

# Load Haar cascades for face and eye detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Parameters for stabilization
BUFFER_SIZE = 10
focus_buffer = deque(maxlen=BUFFER_SIZE)

cap = cv2.VideoCapture(0)

def is_focused(eyes):
    # Simple logic: if both eyes are detected, consider as focused
    return len(eyes) == 2

def get_stable_focus(focus_buffer):
    return sum(focus_buffer) > (BUFFER_SIZE // 2)

while cap.isOpened():
    ret, img = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    focused = False
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]

        eyes = eye_cascade.detectMultiScale(roi_gray)
        focused = is_focused(eyes)
        
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)

    # Update the focus buffer
    focus_buffer.append(focused)
    stable_focus = get_stable_focus(focus_buffer)

    if stable_focus:
        cv2.putText(img, "Focused", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    else:
        cv2.putText(img, "Not Focused", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    cv2.imshow('Focus Detection', img)

    k = cv2.waitKey(10)
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()

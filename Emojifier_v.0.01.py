import cv2
import numpy as np
import pandas as pd
from keras.preprocessing import image
from keras.models import model_from_json
from mtcnn import MTCNN
import emoji

# Load pre-trained facial expression recognition model
model = model_from_json(open("model.json", "r").read())
model.load_weights("model_weights.h5")

# Load the haarcascades for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Define emotions
emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

# Initialize webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

        if np.sum([roi_gray]) != 0:
            roi = roi_gray.astype("float") / 255.0
            roi = image.img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)

            preds = model.predict(roi)[0]
            emotion = emotion_dict[np.argmax(preds)]
            print("Emotion:", emotion)

            # Emoji mapping (simplified)
            if emotion == "Angry":
                emoji_face = emoji.emojize(":angry:", use_aliases=True)
            elif emotion == "Disgusted":
                emoji_face = emoji.emojize(":unamused:", use_aliases=True)
            elif emotion == "Fearful":
                emoji_face = emoji.emojize(":fearful:", use_aliases=True)
            elif emotion == "Happy":
                emoji_face = emoji.emojize(":smile:", use_aliases=True)
            elif emotion == "Neutral":
                emoji_face = emoji.emojize(":neutral_face:", use_aliases=True)
            elif emotion == "Sad":
                emoji_face = emoji.emojize(":cry:", use_aliases=True)
            else:
                emoji_face = emoji.emojize(":open_mouth:", use_aliases=True)

            cv2.putText(frame, emotion, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(frame, emoji_face, (x, y - 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow("Facial Expression Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()


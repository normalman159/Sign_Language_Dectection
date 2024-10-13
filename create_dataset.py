import os
import mediapipe as mp
import cv2
import matplotlib.pyplot as plt
import pickle

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.7)   

height, width = 480, 640

DATA_DIR = './DATA'

data = []
labels = []

def resize_img(img):
    return cv2.resize(img, (width, height))

for dir_ in os.listdir(DATA_DIR):
    print(f"Directory: {dir_}")

    for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):

        data_aux = []
        x_ = []
        y_ = []

        img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
        img = resize_img(img)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        result =  hands.process(img_rgb)

        if result.multi_hand_landmarks: 

            for hand_landmarks in result.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y

                    x_.append(x)
                    y_.append(y)

                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))

            data.append(data_aux)
            labels.append(dir_)


f = open('data.pickle', 'wb')
pickle.dump({'data': data, 'labels': labels}, f)
f.close()

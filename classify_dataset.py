import cv2
import mediapipe as mp
import numpy as np
import pickle

model_dict = pickle.load(open('model.pickle', 'rb'))
model = model_dict['model']

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.5)

expected_features = 84

def hot_key_break():
    if cv2.waitKey(10) & 0xFF == 27:  # Press 'Esc' to exit
        return True
    return False


while True:

    data_aux = []
    x_ = []
    y_ = []

    ret, frame = cap.read()
    
    if not ret:
        break

    H, W, _ = frame.shape

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_rgb_flip = cv2.flip(frame_rgb, 1)
    results = hands.process(frame_rgb_flip)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame_rgb_flip, hand_landmarks, mp_hands.HAND_CONNECTIONS, 
                                      mp_drawing_styles.get_default_hand_landmarks_style(), 
                                      mp_drawing_styles.get_default_hand_connections_style())

        for hand_landmarks in results.multi_hand_landmarks:
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

        x1 = int(min(x_) * W) - 10
        y1 = int(min(y_) * H) - 10

        x2 = int(max(x_) * W) - 10
        y2 = int(max(y_) * H) - 10

        if (len(data_aux) < expected_features):
            data_aux = np.pad(data_aux, (0, expected_features - len(data_aux)), 'constant')
        else:
            data_aux = data_aux[:expected_features]

        prediction_char = model.predict([np.array(data_aux)])
        if len(prediction_char) > 0:
            prediction_char = prediction_char[0]
 
        # Display the predicted character on the frame
        cv2.putText(frame_rgb_flip, prediction_char, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            
    cv2.imshow("Frame", frame_rgb_flip)


    if hot_key_break(): break

cv2.release()
cv2.destroyAllWindows()

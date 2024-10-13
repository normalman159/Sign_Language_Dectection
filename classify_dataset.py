import cv2
import mediapipe as mp
import numpy as np
import pickle
import time

model_dict = pickle.load(open('model.pickle', 'rb'))
model = model_dict['model']

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.5)

expected_features = 84

previous_char = None
time_clear = 4
timeWaitForNewChar = 1
time_in = time.time()
time_previous_char_first = 0
time_del = 0
timeWaitDel = 1
prediction_char = None

def hot_key_break():
    if cv2.waitKey(10) & 0xFF == 27:  # Press 'Esc' to exit
        return True
    return False

def showPredictionPhrase() :
    phrase_text = ''.join(predicted_phrase)
    cv2.rectangle(frame_rgb_flip, (10, 10), (20 + len(phrase_text) * 20, 50), (0, 0, 0), -1)
    cv2.putText(frame_rgb_flip, phrase_text, (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

def deleteChar() :
    if len(predicted_phrase) > 0:
        predicted_phrase.pop()

predicted_phrase = []


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
        time_in = time.time()

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

        #Check if the predicted character is del that pops the last character
        if (prediction_char == 'del') : 
            if (time.time() - time_del) > timeWaitDel:
                deleteChar()
                time_del = time.time()
            previous_char = prediction_char
            time_previous_char_first = time.time()

        # Check if the predicted character is different from the previous one
        elif ((previous_char != prediction_char) or (previous_char is None)) :   
            
            time_current_char = time.time()
            if (time_current_char - time_previous_char_first) > timeWaitForNewChar:
                previous_char = prediction_char
                if (prediction_char == 'space') : prediction_char = ' ' #Check if the prediction_char is space
                predicted_phrase.append(prediction_char)
            
        elif (previous_char == prediction_char) :
            time_previous_char_first = time.time()

        # Display the predicted character on the frame
        cv2.putText(frame_rgb_flip, prediction_char, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Display the predicted phrase in a rectangle
        print(predicted_phrase)

        showPredictionPhrase()

    else: 
        showPredictionPhrase()
        time_previous_char_first = time.time()
        time_out = time.time()
        if (prediction_char == 'del') : deleteChar()
        else : 
            if (time_out - time_in) > time_clear:
                print("Clear")
                predicted_phrase.clear()
                previous_char = None
        
    cv2.imshow("Frame", frame_rgb_flip)


    if hot_key_break(): break

cv2.release()
cv2.destroyAllWindows()

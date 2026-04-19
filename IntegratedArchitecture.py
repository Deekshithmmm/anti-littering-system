from ultralytics import YOLO
import cv2
import time
import math
import pandas as pd
from datetime import datetime
import os
import copy 
import numpy as np 
from deepface import DeepFace
import tensorflow as tf 

# ---------------- YOLO ----------------
model = YOLO('best.pt')
classNames = ['bottle', 'juice-cup', 'nescafe', 'plate', 'tissue']

# ---------------- FINE SYSTEM ----------------
directory = 'FineDatabase'
os.makedirs(directory, exist_ok=True)

excel_path = os.path.join(directory, 'fines.xlsx')

try:
    df = pd.read_excel(excel_path)
except:
    df = pd.DataFrame(columns=["Reg_No", "Date", "Fine"])

def save_dataframe():
    df.to_excel(excel_path, index=False)

def update_fines(name):
    global df
    now = datetime.now()

    if name in df['Reg_No'].values:
        df.loc[df['Reg_No'] == name, 'Fine'] += 200
        df.loc[df['Reg_No'] == name, 'Date'] = now
    else:
        new_row = {"Reg_No": name, "Date": now, "Fine": 200}
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

    save_dataframe()

# ---------------- YOLO DETECTION ----------------
def detect_objects(frame):
    return model(frame, stream=True)

# ---------------- MOVENET ----------------
def run_inference(interpreter, input_size, image):
    image_width, image_height = image.shape[1], image.shape[0]

    input_image = cv2.resize(image, (input_size, input_size))
    input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
    input_image = input_image.reshape(-1, input_size, input_size, 3)
    input_image = tf.cast(input_image, dtype=tf.uint8)

    interpreter.set_tensor(interpreter.get_input_details()[0]['index'], input_image.numpy())
    interpreter.invoke()

    keypoints = interpreter.get_tensor(interpreter.get_output_details()[0]['index'])
    keypoints = np.squeeze(keypoints)

    result = []
    for i in range(17):
        x = int(image_width * keypoints[i][1])
        y = int(image_height * keypoints[i][0])
        result.append([x, y])

    return result

def draw_debug(image, keypoints):
    debug_image = copy.deepcopy(image)

    p1 = keypoints[9]
    p2 = keypoints[10]

    cv2.circle(debug_image, p1, 6, (255, 255, 255), -1)
    cv2.circle(debug_image, p2, 6, (255, 255, 255), -1)

    return debug_image, p1, p2

def calculate_lengths(p1, p2, x1, y1):
    l1 = math.hypot(p1[0]-x1, p1[1]-y1)
    l2 = math.hypot(p2[0]-x1, p2[1]-y1)
    return l1, l2

# ---------------- FACE + FINE ----------------
def detect_fines(frame, length, threshold, people):

    for person_df in people:
        if person_df.empty:
            continue

        person = person_df.iloc[0]

        try:
            x = int(person['source_x'])
            y = int(person['source_y'])
            w = int(person['source_w'])
            h = int(person['source_h'])
        except:
            continue

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        try:
            name = person['identity'].split('\\')[-2]
        except:
            name = "Unknown"

        cv2.putText(frame, name, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        if length > threshold:
            cv2.putText(frame, "LITTERING DETECTED", (x, y-30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            if name != "Unknown":
                update_fines(name)

# ---------------- CAMERA ----------------
cap = cv2.VideoCapture(0)

# ---------------- MAIN ----------------
def main():
    print("[INFO] Starting Stream...")

    interpreter = tf.lite.Interpreter(
        model_path='MoveNet/tflite/lite-model_movenet_singlepose_lightning_tflite_float16_4.tflite'
    )
    interpreter.allocate_tensors()

    last_face_check = 0
    people = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        predictions = detect_objects(frame)

        # 🔥 OPTIMIZATION: run face recognition every 3 seconds
        current_time = time.time()
        if current_time - last_face_check > 3:
            people = DeepFace.find(
                img_path=frame,
                db_path="face_database/",
                model_name="Facenet512",
                enforce_detection=False
            )
            last_face_check = current_time

        for r in predictions:
            for box in r.boxes:

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                keypoints = run_inference(interpreter, 192, frame)
                frame, p1, p2 = draw_debug(frame, keypoints)

                l1, l2 = calculate_lengths(p1, p2, x1, y1)

                if l2 > l1:
                    detect_fines(frame, l1, 100, people)
                else:
                    detect_fines(frame, l2, 100, people)

        cv2.imshow("Anti-Littering System", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
import argparse
import cv2
import numpy as np
import mediapipe as mp
import pickle
import os
import joblib

def extract_landmarks_live(image_bgr, hands):
    """Extract 3D hand landmarks from a live webcam image."""
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    res = hands.process(image_rgb)
    if not res.multi_hand_landmarks:
        return None, res
    
    lm = res.multi_hand_landmarks[0]
    coords = []
    for p in lm.landmark:
        coords.extend([p.x, p.y, p.z])
    
    return np.array(coords, dtype=np.float32).reshape(1, -1), res

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", type=str, default="models/hand_gesture_model.pkl")
    ap.add_argument("--camera", type=int, default=0)
    args = ap.parse_args()

    # Load the model and class names from the single pickle file
    with open(args.model_path, "rb") as f:
        model_data = pickle.load(f)
    
    clf = model_data["model"]
    classes = model_data["classes"]

    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils
    mp_style = mp.solutions.drawing_styles

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError("Could not open camera")

    with mp_hands.Hands(static_image_mode=False, max_num_hands=1, model_complexity=1,
                         min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            
            feat, res = extract_landmarks_live(frame, hands)
            label_text = "No hand"
            
            if feat is not None:
                # Corrected logic: Use the predicted label directly
                predicted_label = clf.predict(feat)[0]
                prob = max(clf.predict_proba(feat)[0])
                
                label_text = f"{predicted_label} ({prob*100:.1f}%)"

            # Draw landmarks
            if res.multi_hand_landmarks:
                for hand_landmarks in res.multi_hand_landmarks:
                    mp_draw.draw_landmarks(
                        frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                        mp_style.get_default_hand_landmarks_style(),
                        mp_style.get_default_hand_connections_style(),
                    )

            cv2.rectangle(frame, (10, 10), (330, 60), (0, 0, 0), -1)
            cv2.putText(frame, label_text, (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
            cv2.imshow("Hand Gesture — Realtime", frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
import os
import pickle
from tqdm import tqdm
import cv2
import mediapipe as mp
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.5
)

def extract_landmarks(image):
    """Extract 3D hand landmarks from an image."""
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    if results.multi_hand_landmarks:
        landmarks = []
        for lm in results.multi_hand_landmarks[0].landmark:
            landmarks.extend([lm.x, lm.y, lm.z])
        return landmarks
    return None

def load_dataset(data_dir, cache_file):
    """Load dataset, extract landmarks, and cache them."""
    if not os.path.exists(data_dir):
        print(f"[ERROR] Dataset directory not found: {data_dir}")
        return [], [], []

    if os.path.exists(cache_file):
        print(f"[INFO] Loading existing cache from {cache_file}")
        with open(cache_file, "rb") as f:
            cache = pickle.load(f)
    else:
        cache = {}

    X, y, class_names = [], [], []
    missing_images = 0

    for label in os.listdir(data_dir):
        label_path = os.path.join(data_dir, label)
        if not os.path.isdir(label_path):
            continue
        
        if label not in class_names:
            class_names.append(label)

        for img_name in tqdm(os.listdir(label_path), desc=f"Processing {label}"):
            img_path = os.path.join(label_path, img_name)

            if not img_name.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            if img_path in cache:
                X.append(cache[img_path])
                y.append(label)
                continue

            image = cv2.imread(img_path)
            if image is None:
                print(f"[WARNING] Could not read {img_path}")
                missing_images += 1
                continue

            feats = extract_landmarks(image)
            if feats:
                X.append(feats)
                y.append(label)
                cache[img_path] = feats
            else:
                missing_images += 1

    with open(cache_file, "wb") as f:
        pickle.dump(cache, f)

    print(f"[INFO] Cache updated. Missing images: {missing_images}")
    return X, y, class_names

if __name__ == "__main__":
    # The base path to the directory containing all the numbered folders (00, 01, etc.)
    data_base_dir = os.path.join("data", "leapGestRecog", "leapGestRecog")
    
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)
    cache_file = os.path.join(model_dir, "landmark_cache.pkl")
    model_path = os.path.join(model_dir, "hand_gesture_model.pkl")

    # Clear the old cache to force the script to re-process all the data
    if os.path.exists(cache_file):
        os.remove(cache_file)
        print("[INFO] Cleared old cache file.")

    # Loop through all 10 folders (00 to 09) and load their data
    X_all, y_all, class_names_all = [], [], []
    for i in range(10): 
        data_dir = os.path.join(data_base_dir, f"{i:02d}")
        print(f"\n[INFO] Loading data from: {data_dir}")
        X_part, y_part, class_names_part = load_dataset(data_dir, cache_file)
        
        X_all.extend(X_part)
        y_all.extend(y_part)
        if not class_names_all:
            class_names_all = class_names_part
    
    print(f"\n[INFO] Processed {len(X_all)} samples across {len(class_names_all)} classes.")

    if len(X_all) == 0:
        print("[ERROR] No landmarks were processed. Check your dataset path and images.")
    else:
        # Train model using all the data
        print("[INFO] Training model...")
        X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.2, random_state=42)
        
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X_train, y_train)

        # Evaluate model
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"[INFO] Model accuracy: {acc*100:.2f}%")

        # Save model
        with open(model_path, "wb") as f:
            pickle.dump({"model": clf, "classes": class_names_all}, f)
        print(f"[INFO] Model saved to {model_path}")
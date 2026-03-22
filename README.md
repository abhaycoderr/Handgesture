# Hand Gesture Recognition — Starter Kit (Landmarks + Classical ML)

This starter kit uses **MediaPipe Hands** to extract 21 hand landmarks from images or webcam frames, converts them into numeric features, and trains a lightweight **scikit-learn** classifier.

## Folder layout you need

```
data/
  fist/
    img001.jpg
    img002.jpg
    ...
  palm/
    ...
  thumbs_up/
    ...
  ok/
    ...
```

Each subfolder name is the gesture label. PNG/JPG/JPEG images are supported.

## Quickstart

1. Create a virtual env (recommended) and install deps:

```bash
pip install -r requirements.txt
```

2. Train with your dataset (update `--data_dir` if different):

```bash
python train_landmark_classifier.py --data_dir data --model_dir models
```

3. Run the real‑time demo (webcam):

```bash
python realtime_infer.py --model_dir models
```

If your camera index isn't 0, pass `--camera 1` (or other index).

## Notes

- Images where no hand is detected are skipped during training.
- Features are 21*3 = 63 values (x, y, z) normalized to the image size by MediaPipe, plus an optional "handedness" score (0…1) for right-hand confidence.
- This approach trains fast and works well with small datasets. If you need higher accuracy on raw pixels, fine-tune a CNN (e.g., MobileNetV3) later.

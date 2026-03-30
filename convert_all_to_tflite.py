"""
convert_all_to_tflite.py

Converts all user .h5 models in the models/ directory to .tflite,
skipping any that already have a .tflite file.
"""

import os
import glob
import tensorflow as tf
from tensorflow.keras.models import load_model

MODELS_DIR = "models"

h5_files = sorted(glob.glob(os.path.join(MODELS_DIR, "user_*_model.h5")))

if not h5_files:
    print("No user .h5 model files found in models/")
    exit()

print(f"Found {len(h5_files)} user .h5 models\n")

converted, skipped = 0, 0

for h5_path in h5_files:
    tflite_path = h5_path.replace(".h5", ".tflite")

    if os.path.exists(tflite_path):
        print(f"  [skip] {os.path.basename(tflite_path)} already exists")
        skipped += 1
        continue

    print(f"  Converting: {os.path.basename(h5_path)} ...", end=" ", flush=True)
    model = load_model(h5_path)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    with open(tflite_path, "wb") as f:
        f.write(tflite_model)

    size_kb = len(tflite_model) / 1024
    print(f"saved ({size_kb:.1f} KB)")
    converted += 1

print(f"\nDone. Converted: {converted}, Skipped: {skipped}")

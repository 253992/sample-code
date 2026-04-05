"""
convert_all_to_tflite.py

Converts all user .h5 models in the models/ directory to .tflite,
skipping any that already have a .tflite file.
"""

import os
import glob
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import LSTM, Input
from tensorflow.keras import Model

MODELS_DIR = "models"


def build_inference_model(trained_model):
    """
    Wrap a trained model with explicit LSTM state I/O for session-aware TFLite inference.

    TFLite inputs:
      sequence_input  (SEQ_LENGTH, num_features)
      state_h         (LSTM_UNITS,)  — hidden state, zeros at session start
      state_c         (LSTM_UNITS,)  — cell state,   zeros at session start

    TFLite outputs:
      output          (NUM_CLASSES,) — fatigue probabilities
      state_h_out     (LSTM_UNITS,)  — pass back as state_h next call
      state_c_out     (LSTM_UNITS,)  — pass back as state_c next call
    """
    lstm_units = trained_model.get_layer('lstm').units
    _, seq_len, num_features = trained_model.input_shape

    seq_input  = Input(shape=(seq_len, num_features), name='sequence_input')
    state_h_in = Input(shape=(lstm_units,), name='state_h')
    state_c_in = Input(shape=(lstm_units,), name='state_c')

    x = trained_model.get_layer('conv1d_1')(seq_input)
    x = trained_model.get_layer('bn_1')(x)
    x = trained_model.get_layer('conv1d_2')(x)
    x = trained_model.get_layer('bn_2')(x)

    # unroll=True forces static unrolling — required for TFLite (no TensorListReserve)
    lstm_stateful = LSTM(lstm_units, return_state=True, unroll=True, name='lstm_stateful')
    lstm_out, state_h_out, state_c_out = lstm_stateful(
        x, initial_state=[state_h_in, state_c_in]
    )

    x = trained_model.get_layer('dense_1')(lstm_out)
    output = trained_model.get_layer('output')(x)

    inference_model = Model(
        inputs=[seq_input, state_h_in, state_c_in],
        outputs=[output, state_h_out, state_c_out],
    )

    inference_model.get_layer('lstm_stateful').set_weights(
        trained_model.get_layer('lstm').get_weights()
    )

    return inference_model


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
    trained_model = load_model(h5_path)
    inference_model = build_inference_model(trained_model)

    converter = tf.lite.TFLiteConverter.from_keras_model(inference_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    with open(tflite_path, "wb") as f:
        f.write(tflite_model)

    size_kb = len(tflite_model) / 1024
    print(f"saved ({size_kb:.1f} KB)")
    converted += 1

print(f"\nDone. Converted: {converted}, Skipped: {skipped}")

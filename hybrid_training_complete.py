"""
FITGUARD HYBRID FATIGUE PREDICTION MODEL

MONTERO, VILLALON, VINLUAN 2026
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import Sequential, Model, load_model
from keras.layers import (
    Conv1D, MaxPooling1D, LSTM,
    Dense, Dropout, BatchNormalization, Input
)
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam
from keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
import pickle
import json
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os

# Create directories for saving models and results
os.makedirs('models', exist_ok=True)
os.makedirs('results', exist_ok=True)
os.makedirs('scalers', exist_ok=True)

# =============================================================================
# CONFIGURATION
# =============================================================================

class Config:
    """Model and training configuration"""

    # --- Sequence construction ---
    # Each row in the CSV is already a ~48-second window summary.
    # SEQ_LENGTH consecutive windows form one training sample.
    # With ~69s between windows, SEQ_LENGTH=5 covers ~5-6 minutes.
    SEQ_LENGTH = 5
    SEQ_OVERLAP = 2  # Overlapping windows between sequences for data augmentation

    # --- Classification ---
    # Binary: 0=Low fatigue (RPE 0-4), 1=High fatigue (RPE 5-10)
    # The model outputs P(High), which is mapped to 4 app-facing levels
    # using FATIGUE_THRESHOLDS in the Android app.
    NUM_CLASSES = 2
    BINARY_LABEL_MAP = {0: 0, 1: 0, 2: 1, 3: 1}  # Original 4-class → binary

    # App-side thresholds: P(High) → 4 fatigue levels for display
    # These are saved to scaler_params.json for Android to use
    FATIGUE_THRESHOLDS = {
        'mild_max':     0.25,  # P(High) < 0.25 → Mild (level 0)
        'moderate_max': 0.50,  # P(High) < 0.50 → Moderate (level 1)
        'high_max':     0.75,  # P(High) < 0.75 → High (level 2)
                               # P(High) >= 0.75 → Critical (level 3)
    }

    # --- Feature columns (19 PPG-only inputs) ---
    # Accelerometer and activity features removed: session analysis showed
    # accel_mag_var and cadence_spm are severe outliers (Z = -2.78 / -2.65)
    # between training data and real app sessions, causing erratic predictions.
    # HR, HRV, and SpO2 features are stable across sessions (all within ±1.2σ).
    FEATURE_COLUMNS = [
        # Heart Rate (7)
        "mean_hr_bpm", "hr_std_bpm", "hr_min_bpm", "hr_max_bpm",
        "hr_range_bpm", "hr_slope_bpm_per_s", "nn_quality_ratio",
        # HRV Time-Domain (5)
        "sdnn_ms", "rmssd_ms", "pnn50_pct", "mean_nn_ms", "cv_nn",
        # HRV Frequency-Domain (4)
        "lf_power_ms2", "hf_power_ms2", "lf_hf_ratio", "total_power_ms2",
        # SpO2 (3)
        "spo2_mean_pct", "spo2_min_pct", "spo2_std_pct",
    ]

    # --- Model architecture ---
    CONV_FILTERS = 64
    LSTM_UNITS = 64
    DENSE_UNITS = 32
    DROPOUT_RATE = 0.4

    # --- Training parameters ---
    EPOCHS = 100
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001

    # --- Fine-tuning parameters ---
    FINETUNE_EPOCHS = 20
    FINETUNE_LR = 0.0001

    # --- File paths ---
    DATA_PATH = "augmented_dataset.csv"
    BASE_MODEL_PATH = "models/base_fatigue_model.h5"
    TFLITE_MODEL_PATH = "models/fatigue_model.tflite"


config = Config()


# =============================================================================
# DATA LOADING AND PREPROCESSING
# =============================================================================

def load_and_preprocess_data(filepath, config):
    """
    Load CSV and perform initial preprocessing.

    Args:
        filepath: Path to features CSV
        config: Configuration object

    Returns:
        df: Preprocessed dataframe
        activity_encoder: Fitted LabelEncoder for activity_label
    """
    print("Loading data...")
    df = pd.read_csv(filepath)

    # --- Validate required columns ---
    missing = [c for c in config.FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")

    if 'fatigue_level' not in df.columns:
        raise ValueError("Missing 'fatigue_level' column (target label)")

    # --- Handle user_id ---
    if 'user_id' not in df.columns:
        print("  No user_id column found — treating all data as single user.")
        df['user_id'] = 'default_user'

    # --- Handle session_id for proper sequence grouping ---
    # Extract session from sequence_id if it follows the pattern:
    # session_XXXXX_seq_XXXXX
    if 'session_id' not in df.columns:
        if 'sequence_id' in df.columns:
            extracted = df['sequence_id'].str.extract(r'(session_\d+)')
            if extracted[0].notna().all():
                df['session_id'] = extracted[0]
            else:
                df['session_id'] = 'session_0'
        else:
            df['session_id'] = 'session_0'

    # --- Encode activity labels ---
    activity_encoder = LabelEncoder()
    if 'activity_label' in df.columns:
        df['activity_encoded'] = activity_encoder.fit_transform(df['activity_label'])
        with open('scalers/activity_encoder.pkl', 'wb') as f:
            pickle.dump(activity_encoder, f)
        print(f"  Activities: {list(activity_encoder.classes_)}")
    else:
        print("  No activity_label column — skipping activity encoding.")
        activity_encoder = None

    # --- Sort by user, session, then time ---
    if 'timestamp' in df.columns:
        df = df.sort_values(['user_id', 'session_id', 'timestamp']).reset_index(drop=True)

    # --- Drop rows with NaN labels ---
    before = len(df)
    df = df.dropna(subset=['fatigue_level']).reset_index(drop=True)
    dropped = before - len(df)
    if dropped > 0:
        print(f"  Dropped {dropped} unlabeled rows ({dropped}/{before})")
    df['fatigue_level'] = df['fatigue_level'].astype(int)

    # --- Remap to binary labels ---
    # Original: 0=Mild, 1=Moderate, 2=High, 3=Critical
    # Binary:   0=Low (Mild+Moderate), 1=High (High+Critical)
    df['fatigue_level_original'] = df['fatigue_level']  # Keep original for analysis
    df['fatigue_level'] = df['fatigue_level'].map(config.BINARY_LABEL_MAP)
    print(f"  Remapped to binary: {df['fatigue_level'].value_counts().sort_index().to_dict()}")

    # --- Detect session boundaries from large time gaps ---
    # Gaps > threshold within the same user/session are treated as
    # separate segments. Sequences never cross segment boundaries.
    SESSION_GAP_THRESHOLD_S = 45.0
    df['segment_id'] = 0
    for (uid, sid), group in df.groupby(['user_id', 'session_id']):
        if len(group) < 2:
            continue
        gaps_s = group['timestamp'].diff() / 1000.0
        boundaries = gaps_s > SESSION_GAP_THRESHOLD_S
        seg_ids = boundaries.cumsum().astype(int)
        # Make segment_id unique across groups by combining session + segment
        df.loc[group.index, 'segment_id'] = seg_ids
    # Combine session_id and segment_id for unique grouping
    df['session_segment'] = df['session_id'] + '_seg_' + df['segment_id'].astype(str)

    # --- Summary ---
    print(f"  Loaded {len(df)} window summaries")
    print(f"  Users: {df['user_id'].nunique()}")
    print(f"  Sessions: {df['session_id'].nunique()}")
    print(f"  Fatigue distribution:\n{df['fatigue_level'].value_counts().sort_index().to_string()}")

    return df, activity_encoder


def create_global_scaler(df, config):
    """
    Create and save a global StandardScaler fitted on all data.

    Args:
        df: Dataframe with feature columns
        config: Configuration object

    Returns:
        scaler: Fitted StandardScaler
    """
    print("\nCreating global scaler...")
    scaler = StandardScaler()
    X_raw = df[config.FEATURE_COLUMNS].values
    scaler.fit(X_raw)

    with open('scalers/global_scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

    scaler_params = {
        'mean': scaler.mean_.tolist(),
        'std': scaler.scale_.tolist(),
        'feature_names': config.FEATURE_COLUMNS,
        'num_classes': config.NUM_CLASSES,
        'fatigue_thresholds': config.FATIGUE_THRESHOLDS,
        'fatigue_level_names': ['Mild', 'Moderate', 'High', 'Critical'],
    }
    with open('scalers/scaler_params.json', 'w') as f:
        json.dump(scaler_params, f, indent=2)

    print(f"  Saved global scaler ({len(config.FEATURE_COLUMNS)} features)")
    return scaler


def create_user_scalers(df, config, min_samples=30):
    """
    Create per-user StandardScalers for personalization.

    Args:
        df: Dataframe with feature columns and user_id
        config: Configuration object
        min_samples: Minimum windows needed to create a user scaler

    Returns:
        user_scalers: Dict of {user_id: StandardScaler}
    """
    print("\nCreating user-specific scalers...")
    user_scalers = {}

    for user_id in df['user_id'].unique():
        user_data = df[df['user_id'] == user_id]
        if len(user_data) < min_samples:
            print(f"  Skipping {user_id} ({len(user_data)} samples < {min_samples})")
            continue

        scaler = StandardScaler()
        scaler.fit(user_data[config.FEATURE_COLUMNS].values)
        user_scalers[user_id] = scaler

        # Save individual JSON for Android deployment
        params = {
            'user_id': user_id,
            'feature_names': config.FEATURE_COLUMNS,
            'mean': scaler.mean_.tolist(),
            'std': scaler.scale_.tolist(),
            'num_classes': config.NUM_CLASSES,
            'fatigue_thresholds': config.FATIGUE_THRESHOLDS,
            'fatigue_level_names': ['Mild', 'Moderate', 'High', 'Critical'],
        }
        with open(f'scalers/user_{user_id}_scaler.json', 'w') as f:
            json.dump(params, f, indent=2)

    with open('scalers/user_scalers.pkl', 'wb') as f:
        pickle.dump(user_scalers, f)

    print(f"  Created scalers for {len(user_scalers)} users")
    return user_scalers


# =============================================================================
# SEQUENCE CONSTRUCTION
# =============================================================================

def create_sequences(df, config, scaler):
    """
    Group consecutive window summaries into sequences for the model.

    Each row in df is already a pre-computed window (~48 seconds).
    This function groups SEQ_LENGTH consecutive windows into one
    training sample, respecting session boundaries.

    Args:
        df: Dataframe sorted by user/session/time
        config: Configuration object
        scaler: Fitted StandardScaler for normalization

    Returns:
        X_seq: np.array of shape (num_sequences, SEQ_LENGTH, num_features)
        y_seq: np.array of integer current fatigue labels
        y_future_seq: np.array of integer future fatigue labels (next window)
    """
    print("\nConstructing sequences...")

    num_features = len(config.FEATURE_COLUMNS)
    seq_len = config.SEQ_LENGTH
    step = seq_len - config.SEQ_OVERLAP  # Stride between sequences

    X_sequences = []
    y_labels = []
    y_future_labels = []

    # Group by user + session + segment to avoid crossing boundaries
    groups = df.groupby(['user_id', 'session_segment'])

    for (user_id, session_segment), group in groups:
        # Normalize features
        X_raw = group[config.FEATURE_COLUMNS].values
        X_scaled = scaler.transform(X_raw)
        y_raw = group['fatigue_level'].values

        # Slide window across this session
        for i in range(0, len(X_scaled) - seq_len + 1, step):
            seq_features = X_scaled[i:i + seq_len]
            seq_labels = y_raw[i:i + seq_len]

            X_sequences.append(seq_features)

            # Current label: fatigue level at the END of this sequence.
            y_labels.append(seq_labels[-1])

            # Future label: fatigue level of the NEXT window after this sequence.
            # If we are at the end of the session, repeat the current label
            # (fatigue is assumed to hold steady when no future data exists).
            if i + seq_len < len(y_raw):
                y_future_labels.append(y_raw[i + seq_len])
            else:
                y_future_labels.append(seq_labels[-1])

    X_seq = np.array(X_sequences)
    y_seq = np.array(y_labels)
    y_future_seq = np.array(y_future_labels)

    print(f"  Sequences: {len(X_seq)}")
    print(f"  Shape: {X_seq.shape} → (samples, {seq_len} windows, {num_features} features)")
    print(f"  Current label distribution:  {dict(zip(*np.unique(y_seq, return_counts=True)))}")
    print(f"  Future label distribution:   {dict(zip(*np.unique(y_future_seq, return_counts=True)))}")

    return X_seq, y_seq, y_future_seq


def prepare_training_data(df, config, scaler=None, user_id=None):
    """
    Complete data preparation pipeline.

    Args:
        df: Raw dataframe
        config: Configuration object
        scaler: Pre-fitted scaler (if None, loads global scaler)
        user_id: If provided, uses user-specific scaler

    Returns:
        X_seq: Prepared sequences
        y_encoded: One-hot encoded current labels
        y_seq: Original integer current labels (for stratification)
        y_future_encoded: One-hot encoded future labels
        y_future_seq: Original integer future labels
    """
    print("\nPreparing training data...")

    if scaler is None:
        if user_id is not None:
            with open('scalers/user_scalers.pkl', 'rb') as f:
                user_scalers = pickle.load(f)
            scaler = user_scalers.get(user_id)
            if scaler is None:
                print(f"  No scaler for {user_id}, falling back to global")

        if scaler is None:
            with open('scalers/global_scaler.pkl', 'rb') as f:
                scaler = pickle.load(f)
            print("  Using global scaler")
        else:
            print(f"  Using scaler for user {user_id}")

    X_seq, y_seq, y_future_seq = create_sequences(df, config, scaler)
    y_encoded = to_categorical(y_seq, num_classes=config.NUM_CLASSES)
    y_future_encoded = to_categorical(y_future_seq, num_classes=config.NUM_CLASSES)

    return X_seq, y_encoded, y_seq, y_future_encoded, y_future_seq


# =============================================================================
# MODEL ARCHITECTURE
# =============================================================================

def build_base_model(config):
    """
    Build the CNN-LSTM hybrid model for fatigue classification + forecasting.

    Input shape: (SEQ_LENGTH, num_features)
      - SEQ_LENGTH consecutive window summaries
      - Each window has num_features pre-computed physiological metrics

    Architecture:
      Conv1D → BN → Conv1D → BN → LSTM → Dense(shared)
                                              │            │
                                       current_fatigue  future_fatigue
                                       (what is now)    (what comes next)

    The shared backbone (Conv1D + LSTM) learns temporal patterns useful for
    both classifying the current state and forecasting the next one.
    Two separate Dense output heads are trained simultaneously:
      - current_fatigue: P(High) at the end of the input sequence
      - future_fatigue:  P(High) at the next window (~48-69 s ahead)

    The future head uses half the loss weight of the current head because
    forecasting is inherently noisier than classifying an observed state.

    Args:
        config: Configuration object

    Returns:
        model: Compiled Keras model (two outputs)
    """
    print("\nBuilding base model...")

    num_features = len(config.FEATURE_COLUMNS)

    # --- Input ---
    inputs = Input(shape=(config.SEQ_LENGTH, num_features), name='input')

    # --- Temporal feature extraction ---
    # Conv1D kernel slides across consecutive windows, learning local
    # patterns like "HR rising while HRV drops over 3 windows".
    x = Conv1D(
        config.CONV_FILTERS,
        kernel_size=3,
        activation='relu',
        padding='same',
        name='conv1d_1'
    )(inputs)
    x = BatchNormalization(name='bn_1')(x)

    x = Conv1D(
        config.CONV_FILTERS * 2,
        kernel_size=3,
        activation='relu',
        padding='same',
        name='conv1d_2'
    )(x)
    x = BatchNormalization(name='bn_2')(x)

    # --- Temporal sequence modeling ---
    # LSTM captures longer-range progression across the full sequence.
    x = LSTM(config.LSTM_UNITS, unroll=True, name='lstm')(x)
    x = Dropout(config.DROPOUT_RATE, name='dropout_1')(x)

    # --- Shared representation ---
    # Both output heads branch from this shared dense layer so the
    # backbone is forced to encode the fatigue trajectory, not just
    # the current state.
    shared = Dense(config.DENSE_UNITS, activation='relu', name='dense_shared')(x)
    shared = Dropout(config.DROPOUT_RATE / 2, name='dropout_shared')(shared)

    # --- Output heads ---
    current_out = Dense(
        config.NUM_CLASSES, activation='softmax', name='current_fatigue'
    )(shared)
    future_out = Dense(
        config.NUM_CLASSES, activation='softmax', name='future_fatigue'
    )(shared)

    model = Model(inputs=inputs, outputs=[current_out, future_out])

    model.compile(
        optimizer=Adam(learning_rate=config.LEARNING_RATE),
        loss={
            'current_fatigue': 'categorical_crossentropy',
            'future_fatigue':  'categorical_crossentropy',
        },
        # Future prediction is noisier so it contributes less to the
        # total loss — keeps the backbone optimised for current accuracy.
        loss_weights={
            'current_fatigue': 1.0,
            'future_fatigue':  0.5,
        },
        metrics={
            'current_fatigue': 'accuracy',
            'future_fatigue':  'accuracy',
        },
    )

    print("  Model built successfully")
    model.summary()

    return model


# =============================================================================
# TRAINING
# =============================================================================

def train_base_model(X_train, y_train, y_future_train,
                     X_val, y_val, y_future_val,
                     config, class_weights=None):
    """
    Train the base model on all users' data.

    Args:
        X_train, y_train: Training features and current labels
        y_future_train: One-hot future labels for training
        X_val, y_val: Validation features and current labels
        y_future_val: One-hot future labels for validation
        config: Configuration object
        class_weights: Optional dict for imbalanced classes

    Returns:
        model: Trained model
        history: Training history
    """
    print("\n" + "=" * 70)
    print("TRAINING BASE MODEL")
    print("=" * 70)

    model = build_base_model(config)

    # Keras 3 does not support class_weight or sample_weight for multi-output
    # models — compile_utils.py resolves weights by positional index and raises
    # KeyError: 0 when the structure doesn't match.
    #
    # Workaround: embed sample weights directly in a tf.data.Dataset as the
    # third element of each (x, y, w) tuple.  Keras reads the weight from the
    # dataset without going through compile_utils path resolution.
    if class_weights is not None:
        train_class_indices = np.argmax(y_train, axis=1)
        sw_train = np.array([class_weights[i] for i in train_class_indices],
                            dtype=np.float32)
        train_ds = (
            tf.data.Dataset
            .from_tensor_slices((
                X_train.astype(np.float32),
                {'current_fatigue': y_train.astype(np.float32),
                 'future_fatigue':  y_future_train.astype(np.float32)},
                sw_train,
            ))
            .shuffle(buffer_size=len(X_train))
            .batch(config.BATCH_SIZE)
            .prefetch(tf.data.AUTOTUNE)
        )
    else:
        train_ds = (
            tf.data.Dataset
            .from_tensor_slices((
                X_train.astype(np.float32),
                {'current_fatigue': y_train.astype(np.float32),
                 'future_fatigue':  y_future_train.astype(np.float32)},
            ))
            .shuffle(buffer_size=len(X_train))
            .batch(config.BATCH_SIZE)
            .prefetch(tf.data.AUTOTUNE)
        )

    val_ds = (
        tf.data.Dataset
        .from_tensor_slices((
            X_val.astype(np.float32),
            {'current_fatigue': y_val.astype(np.float32),
             'future_fatigue':  y_future_val.astype(np.float32)},
        ))
        .batch(config.BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )

    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1,
        ),
        ModelCheckpoint(
            config.BASE_MODEL_PATH,
            monitor='val_current_fatigue_accuracy',
            mode='max',
            save_best_only=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=7,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    print(f"\nTraining for up to {config.EPOCHS} epochs...")
    history = model.fit(
        train_ds,
        epochs=config.EPOCHS,
        validation_data=val_ds,
        callbacks=callbacks,
        verbose=1,
    )

    print("\n  Base model training complete!")
    return model, history


def fine_tune_for_user(base_model_path, user_X, user_y, user_y_future, user_id, config):
    """
    Fine-tune the base model for a specific user.

    Freezes the shared backbone (Conv1D + LSTM) and retrains only the
    shared dense layer and both output heads with a low learning rate
    on the user's personal data.

    Args:
        base_model_path: Path to trained base model
        user_X: User's feature sequences (normalized)
        user_y: User's one-hot current labels
        user_y_future: User's one-hot future labels
        user_id: User identifier
        config: Configuration object

    Returns:
        model: Fine-tuned model
        history: Training history
    """
    print(f"\n{'=' * 70}")
    print(f"FINE-TUNING FOR USER: {user_id}")
    print(f"{'=' * 70}")

    model = load_model(base_model_path)

    # Freeze backbone layers (input, conv1d_1, bn_1, conv1d_2, bn_2, lstm,
    # dropout_1). Keep dense_shared, dropout_shared, current_fatigue,
    # future_fatigue trainable (last 4 layers).
    for layer in model.layers[:-4]:
        layer.trainable = False

    model.compile(
        optimizer=Adam(learning_rate=config.FINETUNE_LR),
        loss={
            'current_fatigue': 'categorical_crossentropy',
            'future_fatigue':  'categorical_crossentropy',
        },
        loss_weights={
            'current_fatigue': 1.0,
            'future_fatigue':  0.5,
        },
        metrics={
            'current_fatigue': 'accuracy',
            'future_fatigue':  'accuracy',
        },
    )

    trainable = sum(1 for l in model.layers if l.trainable)
    print(f"  Trainable layers: {trainable}/{len(model.layers)}")

    X_train, X_val, y_train, y_val, yf_train, yf_val = train_test_split(
        user_X, user_y, user_y_future, test_size=0.2, random_state=42,
    )

    user_model_path = f'models/user_{user_id}_model.h5'
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),
        ModelCheckpoint(
            user_model_path,
            monitor='val_current_fatigue_accuracy',
            mode='max',
            save_best_only=True,
            verbose=1,
        ),
    ]

    history = model.fit(
        X_train,
        {'current_fatigue': y_train, 'future_fatigue': yf_train},
        epochs=config.FINETUNE_EPOCHS,
        batch_size=16,
        validation_data=(
            X_val,
            {'current_fatigue': y_val, 'future_fatigue': yf_val},
        ),
        callbacks=callbacks,
        verbose=1,
    )

    print(f"  Fine-tuned model saved: {user_model_path}")
    return model, history


# =============================================================================
# EVALUATION
# =============================================================================

def evaluate_model(model, X_test, y_test, y_future_test, model_name="Model"):
    """
    Evaluate model and generate reports for both output heads.

    Args:
        model: Trained model (two outputs: current_fatigue, future_fatigue)
        X_test: Test features
        y_test: One-hot current labels
        y_future_test: One-hot future labels
        model_name: Label for reports

    Returns:
        results: Dict with accuracy, predictions, etc.
    """
    print(f"\n{'=' * 70}")
    print(f"EVALUATING: {model_name.upper()}")
    print(f"{'=' * 70}")

    # Model returns [current_probs, future_probs]
    outputs = model.predict(X_test, verbose=0)
    y_pred_probs_current = outputs[0]
    y_pred_probs_future  = outputs[1]

    y_pred_current = np.argmax(y_pred_probs_current, axis=1)
    y_pred_future  = np.argmax(y_pred_probs_future,  axis=1)
    y_true_current = np.argmax(y_test,        axis=1)
    y_true_future  = np.argmax(y_future_test, axis=1)

    acc_current = accuracy_score(y_true_current, y_pred_current)
    acc_future  = accuracy_score(y_true_future,  y_pred_future)
    print(f"\n  Current fatigue accuracy: {acc_current:.4f} ({acc_current * 100:.2f}%)")
    print(f"  Future  fatigue accuracy: {acc_future:.4f}  ({acc_future  * 100:.2f}%)")

    class_names = ["Low", "High"]

    # --- Current head report ---
    present_current = sorted(set(y_true_current) | set(y_pred_current))
    present_names_c = [class_names[i] for i in present_current]
    print("\n  [Current] Classification Report:")
    report_str_current = classification_report(
        y_true_current, y_pred_current,
        labels=present_current, target_names=present_names_c, zero_division=0,
    )
    report_dict_current = classification_report(
        y_true_current, y_pred_current,
        labels=present_current, target_names=present_names_c,
        zero_division=0, output_dict=True,
    )
    print(report_str_current)

    cm_current = confusion_matrix(y_true_current, y_pred_current, labels=range(config.NUM_CLASSES))
    print("  [Current] Confusion Matrix:")
    print(cm_current)

    safe_name = model_name.lower().replace(" ", "_")

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm_current, annot=True, fmt='d', cmap='Blues',
        xticklabels=class_names, yticklabels=class_names,
    )
    plt.title(f'Current Fatigue — {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(f'results/{safe_name}_current_confusion_matrix.png')
    plt.close()

    # --- Future head report ---
    present_future = sorted(set(y_true_future) | set(y_pred_future))
    present_names_f = [class_names[i] for i in present_future]
    print("\n  [Future] Classification Report:")
    report_str_future = classification_report(
        y_true_future, y_pred_future,
        labels=present_future, target_names=present_names_f, zero_division=0,
    )
    report_dict_future = classification_report(
        y_true_future, y_pred_future,
        labels=present_future, target_names=present_names_f,
        zero_division=0, output_dict=True,
    )
    print(report_str_future)

    cm_future = confusion_matrix(y_true_future, y_pred_future, labels=range(config.NUM_CLASSES))
    print("  [Future] Confusion Matrix:")
    print(cm_future)

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm_future, annot=True, fmt='d', cmap='Oranges',
        xticklabels=class_names, yticklabels=class_names,
    )
    plt.title(f'Future Fatigue — {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(f'results/{safe_name}_future_confusion_matrix.png')
    plt.close()

    # --- Trend transition summary ---
    # Shows how often the model predicts each current→future combination,
    # giving a picture of the trend insights the app will display.
    fatigue_names = ["Mild", "Moderate", "High", "Critical"]
    thresholds = config.FATIGUE_THRESHOLDS

    def to_4level(p_high_arr):
        return np.where(
            p_high_arr < thresholds['mild_max'], 0,
            np.where(p_high_arr < thresholds['moderate_max'], 1,
                     np.where(p_high_arr < thresholds['high_max'], 2, 3)))

    p_high_current = y_pred_probs_current[:, 1]
    p_high_future  = y_pred_probs_future[:, 1]
    level_current  = to_4level(p_high_current)
    level_future   = to_4level(p_high_future)

    print(f"\n  Trend Transition Distribution (predicted current → predicted future):")
    for cur in range(4):
        for fut in range(4):
            count = np.sum((level_current == cur) & (level_future == fut))
            if count > 0:
                arrow = "→" if cur == fut else ("↑" if fut > cur else "↓")
                print(f"    {fatigue_names[cur]:>10s} {arrow} {fatigue_names[fut]:<10s}: {count:>4d}")

    print(f"\n  P(High) current: mean={p_high_current.mean():.3f}  "
          f"P(High) future: mean={p_high_future.mean():.3f}")

    return {
        'accuracy': acc_current,
        'future_accuracy': acc_future,
        'predictions': y_pred_current,
        'future_predictions': y_pred_future,
        'true_labels': y_true_current,
        'true_future_labels': y_true_future,
        'probabilities': y_pred_probs_current,
        'future_probabilities': y_pred_probs_future,
        'report_str': report_str_current,
        'report_dict': report_dict_current,
        'future_report_str': report_str_future,
        'future_report_dict': report_dict_future,
    }


def plot_training_history(history, title="Training History"):
    """Plot training and validation accuracy/loss curves for both output heads."""
    h = history.history
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Current fatigue accuracy
    axes[0].plot(h['current_fatigue_accuracy'], label='Train')
    axes[0].plot(h['val_current_fatigue_accuracy'], label='Validation')
    axes[0].set_title(f'{title} — Current Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True)

    # Future fatigue accuracy
    axes[1].plot(h['future_fatigue_accuracy'], label='Train')
    axes[1].plot(h['val_future_fatigue_accuracy'], label='Validation')
    axes[1].set_title(f'{title} — Future Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    axes[1].grid(True)

    # Total loss
    axes[2].plot(h['loss'], label='Train')
    axes[2].plot(h['val_loss'], label='Validation')
    axes[2].set_title(f'{title} — Total Loss')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Loss')
    axes[2].legend()
    axes[2].grid(True)

    plt.tight_layout()
    safe_name = title.lower().replace(" ", "_")
    plt.savefig(f'results/{safe_name}.png')
    plt.close()


# =============================================================================
# MODEL EXPORT FOR ANDROID
# =============================================================================

def export_to_tflite(model_path, output_path):
    """
    Convert Keras model to TensorFlow Lite for Android deployment.

    Args:
        model_path: Path to .h5 model
        output_path: Path for .tflite output
    """
    print("\n" + "=" * 70)
    print("CONVERTING TO TENSORFLOW LITE")
    print("=" * 70)

    model = load_model(model_path)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    with open(output_path, 'wb') as f:
        f.write(tflite_model)

    size_kb = len(tflite_model) / 1024
    print(f"  Saved: {output_path} ({size_kb:.1f} KB)")


# =============================================================================
# MAIN TRAINING PIPELINE
# =============================================================================

def main():
    print("\n" + "=" * 70)
    print("FITGUARD FATIGUE PREDICTION — TRAINING PIPELINE")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # -----------------------------------------------------------------
    # STEP 1: Load and preprocess
    # -----------------------------------------------------------------
    df, activity_encoder = load_and_preprocess_data(config.DATA_PATH, config)

    # -----------------------------------------------------------------
    # STEP 2: Create scalers
    # -----------------------------------------------------------------
    global_scaler = create_global_scaler(df, config)
    user_scalers = create_user_scalers(df, config)

    # -----------------------------------------------------------------
    # STEP 3: Build sequences (current + future labels)
    # -----------------------------------------------------------------
    X_seq, y_encoded, y_seq, y_future_encoded, y_future_seq = prepare_training_data(
        df, config, scaler=global_scaler
    )

    if len(X_seq) < 10:
        print("\n" + "!" * 70)
        print("WARNING: Very few sequences created.")
        print(f"  You have {len(df)} window summaries → {len(X_seq)} sequences")
        print(f"  (SEQ_LENGTH={config.SEQ_LENGTH}, overlap={config.SEQ_OVERLAP})")
        print()
        print("  For meaningful training you need at minimum:")
        print("    - 500+ sequences (ideally 2000+)")
        print("    - Multiple fatigue levels represented")
        print("    - Multiple users and sessions")
        print()
        print("  Current data is useful for verifying the pipeline runs,")
        print("  but the model will not learn meaningful patterns.")
        print("!" * 70)

    # -----------------------------------------------------------------
    # STEP 4: Split data (current and future labels split together)
    # -----------------------------------------------------------------
    print("\nSplitting data...")

    # Check if stratification is possible (need 2+ classes with 2+ samples)
    unique_labels, label_counts = np.unique(y_seq, return_counts=True)
    can_stratify = len(unique_labels) > 1 and all(c >= 2 for c in label_counts)

    stratify_arg = y_seq if can_stratify else None
    if not can_stratify:
        print("  Cannot stratify — only one class or too few samples per class.")

    X_train, X_test, y_train, y_test, yf_train, yf_test = train_test_split(
        X_seq, y_encoded, y_future_encoded,
        test_size=0.2,
        random_state=42,
        stratify=stratify_arg,
    )

    # Further split train → train + val
    stratify_train = np.argmax(y_train, axis=1) if can_stratify else None
    X_train, X_val, y_train, y_val, yf_train, yf_val = train_test_split(
        X_train, y_train, yf_train,
        test_size=0.2,
        random_state=42,
        stratify=stratify_train,
    )

    print(f"  Train:      {len(X_train)} sequences")
    print(f"  Validation: {len(X_val)} sequences")
    print(f"  Test:       {len(X_test)} sequences")

    # -----------------------------------------------------------------
    # STEP 5: Compute class weights for imbalanced data
    # -----------------------------------------------------------------
    train_labels = np.argmax(y_train, axis=1)
    unique_train = np.unique(train_labels)

    if len(unique_train) > 1:
        weights = compute_class_weight('balanced', classes=unique_train, y=train_labels)
        class_weights = dict(zip(unique_train, weights))
        print(f"\n  Class weights: {class_weights}")
    else:
        class_weights = None
        print("\n  Single class in training data — no class weighting.")

    # -----------------------------------------------------------------
    # STEP 6: Train base model
    # -----------------------------------------------------------------
    base_model, history = train_base_model(
        X_train, y_train, yf_train,
        X_val, y_val, yf_val,
        config, class_weights,
    )
    plot_training_history(history, "Base Model Training")

    # -----------------------------------------------------------------
    # STEP 7: Evaluate both output heads
    # -----------------------------------------------------------------
    base_results = evaluate_model(base_model, X_test, y_test, yf_test, "Base Model")

    # -----------------------------------------------------------------
    # STEP 8: Export to TFLite
    # -----------------------------------------------------------------
    export_to_tflite(config.BASE_MODEL_PATH, config.TFLITE_MODEL_PATH)

    # -----------------------------------------------------------------
    # STEP 9: Personalization demo (if enough user data)
    # -----------------------------------------------------------------
    print("\n" + "=" * 70)
    print("PERSONALIZATION DEMO")
    print("=" * 70)

    for user_id in df['user_id'].unique():
        user_df = df[df['user_id'] == user_id]

        if len(user_df) < config.SEQ_LENGTH + 20:
            print(f"\n  {user_id}: Not enough data for personalization demo "
                  f"({len(user_df)} windows)")
            continue

        print(f"\n  Personalizing for: {user_id}")

        user_scaler = user_scalers.get(user_id, global_scaler)
        user_X, user_y, user_y_seq, user_yf, _ = prepare_training_data(
            user_df, config, scaler=user_scaler
        )

        if len(user_X) < 20:
            print(f"  Too few sequences ({len(user_X)}), skipping.")
            continue

        personalized_model, ft_history = fine_tune_for_user(
            config.BASE_MODEL_PATH, user_X, user_y, user_yf, user_id, config
        )
        plot_training_history(ft_history, f"Fine-tuning {user_id}")

        # Compare base vs personalized on user data (current head only)
        user_X_train, user_X_test, user_y_train, user_y_test, user_yf_train, user_yf_test = \
            train_test_split(user_X, user_y, user_yf, test_size=0.2, random_state=42)

        base_pred = base_model.predict(user_X_test, verbose=0)
        base_acc = accuracy_score(
            np.argmax(user_y_test, axis=1), np.argmax(base_pred[0], axis=1)
        )

        pers_results = evaluate_model(
            personalized_model, user_X_test, user_y_test, user_yf_test,
            f"Personalized ({user_id})"
        )

        if base_acc > 0:
            improvement = (pers_results['accuracy'] - base_acc) / base_acc * 100
        else:
            improvement = 0.0

        print(f"\n  Base accuracy (current head):         {base_acc:.4f}")
        print(f"  Personalized accuracy (current head): {pers_results['accuracy']:.4f}")
        print(f"  Improvement:                          {improvement:+.2f}%")

    # -----------------------------------------------------------------
    # STEP 10: Save summary
    # -----------------------------------------------------------------
    summary = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'config': {
            'seq_length': config.SEQ_LENGTH,
            'seq_overlap': config.SEQ_OVERLAP,
            'num_features': len(config.FEATURE_COLUMNS),
            'num_classes': config.NUM_CLASSES,
            'classification': 'binary (Low vs High fatigue)',
            'binary_label_map': config.BINARY_LABEL_MAP,
            'fatigue_thresholds': config.FATIGUE_THRESHOLDS,
            'epochs': config.EPOCHS,
            'batch_size': config.BATCH_SIZE,
        },
        'data': {
            'total_windows': len(df),
            'total_sequences': len(X_seq),
            'num_users': df['user_id'].nunique(),
            'num_sessions': df['session_id'].nunique(),
            'train_sequences': len(X_train),
            'val_sequences': len(X_val),
            'test_sequences': len(X_test),
            'feature_columns': config.FEATURE_COLUMNS,
        },
        'base_model': {
            'current_accuracy': float(base_results['accuracy']),
            'future_accuracy':  float(base_results['future_accuracy']),
            'classification_report': base_results['report_dict'],
            'future_classification_report': base_results['future_report_dict'],
            'model_path': config.BASE_MODEL_PATH,
            'tflite_path': config.TFLITE_MODEL_PATH,
        },
    }

    with open('results/training_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    # -----------------------------------------------------------------
    # Final report
    # -----------------------------------------------------------------
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"\n  Files created:")
    print(f"    Base model:        {config.BASE_MODEL_PATH}")
    print(f"    TFLite model:      {config.TFLITE_MODEL_PATH}")
    print(f"    Global scaler:     scalers/global_scaler.pkl")
    print(f"    Scaler params:     scalers/scaler_params.json")
    print(f"    User scalers:      scalers/user_scalers.pkl")
    print(f"    Training summary:  results/training_summary.json")
    print(f"\n  Results:")
    print(f"    Current fatigue accuracy: {base_results['accuracy']:.4f} "
          f"({base_results['accuracy'] * 100:.2f}%)")
    print(f"    Future  fatigue accuracy: {base_results['future_accuracy']:.4f} "
          f"({base_results['future_accuracy'] * 100:.2f}%)")
    print(f"\n  [Current] Classification Report (Base Model):")
    for line in base_results['report_str'].splitlines():
        print(f"    {line}")
    print(f"\n  Next steps for Android deployment:")
    print(f"    1. Copy {config.TFLITE_MODEL_PATH} → app/src/main/assets/")
    print(f"    2. Copy scalers/scaler_params.json → app/src/main/assets/")
    print(f"    3. Implement sequence buffering in Android (collect {config.SEQ_LENGTH}")
    print(f"       consecutive window summaries before running inference)")
    print(f"    4. Model now has TWO output tensors:")
    print(f"         output[0] → current_fatigue [P(Low), P(High)]")
    print(f"         output[1] → future_fatigue  [P(Low), P(High)]")
    print(f"    5. Apply FATIGUE_THRESHOLDS to each P(High) to get 4-level labels:")
    print(f"         P(High) < 0.25 → Mild")
    print(f"         P(High) < 0.50 → Moderate")
    print(f"         P(High) < 0.75 → High")
    print(f"         P(High) >= 0.75 → Critical")
    print(f"    6. Derive trend from current→future level pair for user insight")
    print(f"\n  Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
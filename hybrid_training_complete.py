"""
HYBRID FATIGUE PREDICTION MODEL - COMPLETE TRAINING CODE
========================================================

This script includes:
1. Base model training (on all participants)
2. User-specific normalization
3. Fine-tuning capability
4. Model export for Android deployment

Author: Your Name
Date: 2026
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

    # --- Feature columns (33 model inputs) ---
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
        # Accelerometer (9)
        "accel_x_mean", "accel_y_mean", "accel_z_mean",
        "accel_x_var", "accel_y_var", "accel_z_var",
        "accel_mag_mean", "accel_mag_var", "accel_peak",
        # Skin Temperature (3)
        #"skin_temp_obj", "skin_temp_delta", "skin_temp_ambient",
        # Activity Context (2)
        "total_steps", "cadence_spm",
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
        y_seq: np.array of integer fatigue labels
    """
    print("\nConstructing sequences...")

    num_features = len(config.FEATURE_COLUMNS)
    seq_len = config.SEQ_LENGTH
    step = seq_len - config.SEQ_OVERLAP  # Stride between sequences

    X_sequences = []
    y_labels = []

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

            # Label strategy: use the LAST window's fatigue level.
            # This makes the model predict "what fatigue level is the
            # user at by the end of this sequence?"
            # Alternative: np.max(seq_labels) for escalation detection.
            y_labels.append(seq_labels[-1])

    X_seq = np.array(X_sequences)
    y_seq = np.array(y_labels)

    print(f"  Sequences: {len(X_seq)}")
    print(f"  Shape: {X_seq.shape} → (samples, {seq_len} windows, {num_features} features)")
    print(f"  Label distribution: {dict(zip(*np.unique(y_seq, return_counts=True)))}")

    return X_seq, y_seq


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
        y_encoded: One-hot encoded labels
        y_seq: Original integer labels (for stratification)
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

    X_seq, y_seq = create_sequences(df, config, scaler)
    y_encoded = to_categorical(y_seq, num_classes=config.NUM_CLASSES)

    return X_seq, y_encoded, y_seq


# =============================================================================
# MODEL ARCHITECTURE
# =============================================================================

def build_base_model(config):
    """
    Build the CNN-LSTM hybrid model for fatigue prediction.

    Input shape: (SEQ_LENGTH, num_features)
      - SEQ_LENGTH consecutive window summaries
      - Each window has num_features pre-computed physiological metrics

    Architecture:
      Conv1D → BN → Conv1D → BN → LSTM → Dense → Softmax

    Conv1D slides across the temporal axis (consecutive windows),
    learning local patterns like "HR rising while HRV drops over
    3 consecutive windows". LSTM then captures longer-range
    progression across the full sequence.

    Args:
        config: Configuration object

    Returns:
        model: Compiled Keras model
    """
    print("\nBuilding base model...")

    num_features = len(config.FEATURE_COLUMNS)

    model = Sequential([
        # --- Temporal feature extraction ---
        # Conv1D kernel slides across consecutive windows
        Conv1D(
            config.CONV_FILTERS,
            kernel_size=3,
            activation='relu',
            padding='same',
            input_shape=(config.SEQ_LENGTH, num_features),
            name='conv1d_1'
        ),
        BatchNormalization(name='bn_1'),

        Conv1D(
            config.CONV_FILTERS * 2,
            kernel_size=3,
            activation='relu',
            padding='same',
            name='conv1d_2'
        ),
        BatchNormalization(name='bn_2'),

        # --- Temporal sequence modeling ---
        LSTM(config.LSTM_UNITS, unroll=True, name='lstm'),
        Dropout(config.DROPOUT_RATE, name='dropout_1'),

        # --- Classification head ---
        Dense(config.DENSE_UNITS, activation='relu', name='dense_1'),
        Dropout(config.DROPOUT_RATE / 2, name='dropout_2'),
        Dense(config.NUM_CLASSES, activation='softmax', name='output'),
    ])

    model.compile(
        optimizer=Adam(learning_rate=config.LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy'],
    )

    print("  Model built successfully")
    model.summary()

    return model


# =============================================================================
# TRAINING
# =============================================================================

def train_base_model(X_train, y_train, X_val, y_val, config, class_weights=None):
    """
    Train the base model on all users' data.

    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data
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

    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1,
        ),
        ModelCheckpoint(
            config.BASE_MODEL_PATH,
            monitor='val_accuracy',
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
        X_train, y_train,
        epochs=config.EPOCHS,
        batch_size=config.BATCH_SIZE,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1,
    )

    print("\n  Base model training complete!")
    return model, history


def fine_tune_for_user(base_model_path, user_X, user_y, user_id, config):
    """
    Fine-tune the base model for a specific user.

    Freezes early layers and retrains only the classification head
    with a low learning rate on the user's personal data.

    Args:
        base_model_path: Path to trained base model
        user_X: User's feature sequences (normalized)
        user_y: User's one-hot labels
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

    # Freeze everything except the last 3 layers (dense_1, dropout_2, output)
    for layer in model.layers[:-3]:
        layer.trainable = False

    model.compile(
        optimizer=Adam(learning_rate=config.FINETUNE_LR),
        loss='categorical_crossentropy',
        metrics=['accuracy'],
    )

    trainable = sum(1 for l in model.layers if l.trainable)
    print(f"  Trainable layers: {trainable}/{len(model.layers)}")

    X_train, X_val, y_train, y_val = train_test_split(
        user_X, user_y, test_size=0.2, random_state=42,
    )

    user_model_path = f'models/user_{user_id}_model.h5'
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),
        ModelCheckpoint(user_model_path, monitor='val_accuracy', save_best_only=True, verbose=1),
    ]

    history = model.fit(
        X_train, y_train,
        epochs=config.FINETUNE_EPOCHS,
        batch_size=16,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1,
    )

    print(f"  Fine-tuned model saved: {user_model_path}")
    return model, history


# =============================================================================
# EVALUATION
# =============================================================================

def evaluate_model(model, X_test, y_test, model_name="Model"):
    """
    Evaluate model and generate reports.

    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels (one-hot)
        model_name: Label for reports

    Returns:
        results: Dict with accuracy, predictions, etc.
    """
    print(f"\n{'=' * 70}")
    print(f"EVALUATING: {model_name.upper()}")
    print(f"{'=' * 70}")

    y_pred_probs = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = np.argmax(y_test, axis=1)

    accuracy = accuracy_score(y_true, y_pred)
    print(f"\n  Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")

    class_names = ["Low", "High"]

    # Only report on classes that appear in the data
    present_classes = sorted(set(y_true) | set(y_pred))
    present_names = [class_names[i] for i in present_classes]

    print("\n  Classification Report:")
    report_str = classification_report(
        y_true, y_pred,
        labels=present_classes,
        target_names=present_names,
        zero_division=0,
    )
    report_dict = classification_report(
        y_true, y_pred,
        labels=present_classes,
        target_names=present_names,
        zero_division=0,
        output_dict=True,
    )
    print(report_str)

    cm = confusion_matrix(y_true, y_pred, labels=range(config.NUM_CLASSES))
    print("  Confusion Matrix:")
    print(cm)

    # Plot confusion matrix
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=class_names, yticklabels=class_names,
    )
    plt.title(f'Confusion Matrix — {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    safe_name = model_name.lower().replace(" ", "_")
    plt.savefig(f'results/{safe_name}_confusion_matrix.png')
    plt.close()

    # Show 4-level fatigue distribution from P(High) probabilities
    if y_pred_probs.shape[1] == 2:
        p_high = y_pred_probs[:, 1]
        thresholds = config.FATIGUE_THRESHOLDS
        fatigue_4class = np.where(
            p_high < thresholds['mild_max'], 0,
            np.where(p_high < thresholds['moderate_max'], 1,
                     np.where(p_high < thresholds['high_max'], 2, 3)))
        fatigue_names = ["Mild", "Moderate", "High", "Critical"]
        unique_4, counts_4 = np.unique(fatigue_4class, return_counts=True)
        print(f"\n  4-Level Fatigue Distribution (from P(High) thresholds):")
        for u, c in zip(unique_4, counts_4):
            print(f"    {fatigue_names[u]:>10s}: {c:>4d} ({c / len(fatigue_4class) * 100:.1f}%)")
        print(f"  P(High) stats: mean={p_high.mean():.3f}, min={p_high.min():.3f}, max={p_high.max():.3f}")

    return {
        'accuracy': accuracy,
        'predictions': y_pred,
        'true_labels': y_true,
        'probabilities': y_pred_probs,
        'report_str': report_str,
        'report_dict': report_dict,
    }


def plot_training_history(history, title="Training History"):
    """Plot training and validation accuracy/loss curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(history.history['accuracy'], label='Train')
    ax1.plot(history.history['val_accuracy'], label='Validation')
    ax1.set_title(f'{title} — Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True)

    ax2.plot(history.history['loss'], label='Train')
    ax2.plot(history.history['val_loss'], label='Validation')
    ax2.set_title(f'{title} — Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)

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
    # STEP 2: Split users into train / val / test (user-based, no leakage)
    # -----------------------------------------------------------------
    all_users = df['user_id'].unique()
    rng = np.random.default_rng(42)
    shuffled_users = rng.permutation(all_users)

    n = len(shuffled_users)
    n_test = max(1, round(n * 0.2))
    n_val  = max(1, round(n * 0.2))

    test_users  = shuffled_users[:n_test]
    val_users   = shuffled_users[n_test:n_test + n_val]
    train_users = shuffled_users[n_test + n_val:]

    print(f"\nUser-based split ({n} users total):")
    print(f"  Train: {len(train_users)} users — {list(train_users)}")
    print(f"  Val:   {len(val_users)} users — {list(val_users)}")
    print(f"  Test:  {len(test_users)} users — {list(test_users)}")

    df_train = df[df['user_id'].isin(train_users)].reset_index(drop=True)
    df_val   = df[df['user_id'].isin(val_users)].reset_index(drop=True)
    df_test  = df[df['user_id'].isin(test_users)].reset_index(drop=True)

    # -----------------------------------------------------------------
    # STEP 3: Create scalers (fit on training users only)
    # -----------------------------------------------------------------
    global_scaler = create_global_scaler(df_train, config)
    user_scalers  = create_user_scalers(df_train, config)

    # -----------------------------------------------------------------
    # STEP 4: Build sequences per split
    # -----------------------------------------------------------------
    print("\nBuilding sequences...")
    X_train, y_train, _ = prepare_training_data(df_train, config, scaler=global_scaler)
    X_val,   y_val,   _ = prepare_training_data(df_val,   config, scaler=global_scaler)
    X_test,  y_test,  _ = prepare_training_data(df_test,  config, scaler=global_scaler)

    print(f"  Train:      {len(X_train)} sequences ({len(train_users)} users)")
    print(f"  Validation: {len(X_val)} sequences ({len(val_users)} users)")
    print(f"  Test:       {len(X_test)} sequences ({len(test_users)} users)")

    if len(X_train) < 10:
        print("\n" + "!" * 70)
        print("WARNING: Very few training sequences created.")
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
        X_train, y_train, X_val, y_val, config, class_weights
    )
    plot_training_history(history, "Base Model Training")

    # -----------------------------------------------------------------
    # STEP 7: Evaluate
    # -----------------------------------------------------------------
    base_results = evaluate_model(base_model, X_test, y_test, "Base Model")

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

    for user_id in train_users:
        user_df = df_train[df_train['user_id'] == user_id]

        if len(user_df) < config.SEQ_LENGTH + 20:
            print(f"\n  {user_id}: Not enough data for personalization demo "
                  f"({len(user_df)} windows)")
            continue

        print(f"\n  Personalizing for: {user_id}")

        user_scaler = user_scalers.get(user_id, global_scaler)
        user_X, user_y, user_y_seq = prepare_training_data(
            user_df, config, scaler=user_scaler
        )

        if len(user_X) < 20:
            print(f"  Too few sequences ({len(user_X)}), skipping.")
            continue

        personalized_model, ft_history = fine_tune_for_user(
            config.BASE_MODEL_PATH, user_X, user_y, user_id, config
        )
        plot_training_history(ft_history, f"Fine-tuning {user_id}")

        # Compare base vs personalized on user data
        user_X_train, user_X_test, user_y_train, user_y_test = train_test_split(
            user_X, user_y, test_size=0.2, random_state=42,
        )

        base_pred = base_model.predict(user_X_test, verbose=0)
        base_acc = accuracy_score(np.argmax(user_y_test, axis=1), np.argmax(base_pred, axis=1))

        pers_results = evaluate_model(
            personalized_model, user_X_test, user_y_test,
            f"Personalized ({user_id})"
        )

        if base_acc > 0:
            improvement = (pers_results['accuracy'] - base_acc) / base_acc * 100
        else:
            improvement = 0.0

        print(f"\n  Base accuracy:         {base_acc:.4f}")
        print(f"  Personalized accuracy: {pers_results['accuracy']:.4f}")
        print(f"  Improvement:           {improvement:+.2f}%")

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
            'total_sequences': len(X_train) + len(X_val) + len(X_test),
            'num_users': df['user_id'].nunique(),
            'num_sessions': df['session_id'].nunique(),
            'train_users': len(train_users),
            'val_users': len(val_users),
            'test_users': len(test_users),
            'train_sequences': len(X_train),
            'val_sequences': len(X_val),
            'test_sequences': len(X_test),
            'feature_columns': config.FEATURE_COLUMNS,
        },
        'base_model': {
            'accuracy': float(base_results['accuracy']),
            'classification_report': base_results['report_dict'],
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
    print(f"    Base model accuracy: {base_results['accuracy']:.4f} "
          f"({base_results['accuracy'] * 100:.2f}%)")
    print(f"\n  Classification Report (Base Model):")
    for line in base_results['report_str'].splitlines():
        print(f"    {line}")
    print(f"\n  Next steps for Android deployment:")
    print(f"    1. Copy {config.TFLITE_MODEL_PATH} → app/src/main/assets/")
    print(f"    2. Copy scalers/scaler_params.json → app/src/main/assets/")
    print(f"    3. Implement sequence buffering in Android (collect {config.SEQ_LENGTH}")
    print(f"       consecutive window summaries before running inference)")
    print(f"    4. Model outputs [P(Low), P(High)] — use P(High) to determine")
    print(f"       4 fatigue levels using thresholds in scaler_params.json:")
    print(f"         P(High) < 0.25 → Mild")
    print(f"         P(High) < 0.50 → Moderate")
    print(f"         P(High) < 0.75 → High")
    print(f"         P(High) >= 0.75 → Critical")
    print(f"\n  Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
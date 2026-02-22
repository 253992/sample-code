"""
PERSONALIZATION WORKFLOW
========================

This script personalizes the base fatigue model for individual users.
Run AFTER training the base model with hybrid_training_complete.py.

Usage:
    1. Train base model:   python hybrid_training_complete.py
    2. Personalize:        python personalization_workflow.py --user_data user_features.csv --user_id user_01

The workflow:
    - Loads the base model and global scaler from Phase 1
    - Creates a user-specific scaler from the user's calibration data
    - Compares 3 approaches: global norm, user norm, fine-tuning
    - Exports the best personalized model as TFLite for Android

Input:  User's features.csv from a calibration session (~30-40 min workout)
Output: models/user_{id}_model.h5, scalers/user_{id}_scaler.json
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
import json
import argparse
import os

os.makedirs('models', exist_ok=True)
os.makedirs('scalers', exist_ok=True)
os.makedirs('results', exist_ok=True)


# =============================================================================
# SHARED CONFIGURATION (must match hybrid_training_complete.py)
# =============================================================================

class Config:
    """Must stay in sync with the base training config."""

    SEQ_LENGTH = 5
    SEQ_OVERLAP = 2
    NUM_CLASSES = 4  # 0=Mild, 1=Moderate, 2=High, 3=Critical

    # 30 features matching the current CSV schema (35 cols - 5 metadata/label cols)
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
        # Activity Context (2)
        "total_steps", "cadence_spm",
    ]

    # Session boundary: gaps larger than this (seconds) split segments
    SESSION_GAP_THRESHOLD_S = 45.0

    # Fine-tuning
    FINETUNE_EPOCHS = 20
    FINETUNE_LR = 0.0001
    FINETUNE_BATCH_SIZE = 16

    # Paths (outputs from hybrid_training_complete.py)
    BASE_MODEL_PATH = "models/base_fatigue_model.h5"
    GLOBAL_SCALER_PATH = "scalers/global_scaler.pkl"
    USER_SCALERS_PATH = "scalers/user_scalers.pkl"


config = Config()


# =============================================================================
# DATA PREPARATION (shared logic with hybrid_training)
# =============================================================================

def load_user_data(filepath, config):
    """
    Load and preprocess a single user's features.csv.

    Handles: sorting by timestamp, dropping NaN labels,
    detecting session boundaries from large time gaps.

    Args:
        filepath: Path to the user's features CSV
        config: Configuration object

    Returns:
        df: Cleaned, sorted dataframe
    """
    print(f"Loading user data from: {filepath}")
    df = pd.read_csv(filepath)

    # Validate columns
    missing = [c for c in config.FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")

    # Sort by timestamp
    if 'timestamp' in df.columns:
        df = df.sort_values('timestamp').reset_index(drop=True)

    # Drop rows with NaN labels (unlabeled windows)
    before = len(df)
    df = df.dropna(subset=['fatigue_level']).reset_index(drop=True)
    dropped = before - len(df)
    if dropped > 0:
        print(f"  Dropped {dropped} unlabeled rows ({dropped}/{before})")

    # Convert fatigue_level to int
    df['fatigue_level'] = df['fatigue_level'].astype(int)

    # Detect session boundaries from large time gaps
    if 'timestamp' in df.columns and len(df) > 1:
        gaps_s = df['timestamp'].diff() / 1000.0
        boundary_mask = gaps_s > config.SESSION_GAP_THRESHOLD_S
        # Assign segment IDs: increment at each boundary
        df['segment_id'] = boundary_mask.cumsum().astype(int)
        n_segments = df['segment_id'].nunique()
        if n_segments > 1:
            print(f"  Detected {n_segments} segments (gaps > {config.SESSION_GAP_THRESHOLD_S}s)")
            for seg_id in df['segment_id'].unique():
                seg = df[df['segment_id'] == seg_id]
                dur = (seg['timestamp'].iloc[-1] - seg['timestamp'].iloc[0]) / 1000
                print(f"    Segment {seg_id}: {len(seg)} rows, {dur:.0f}s")
    else:
        df['segment_id'] = 0

    print(f"  Rows: {len(df)}")
    print(f"  Fatigue levels: {df['fatigue_level'].value_counts().sort_index().to_dict()}")
    print(f"  RPE values: {df['rpe_raw'].value_counts().sort_index().to_dict()}")

    return df


def create_sequences_from_segments(df, config, scaler):
    """
    Group consecutive windows into sequences, respecting segment boundaries.

    Segments are continuous blocks of data separated by large time gaps.
    Sequences never cross segment boundaries.

    Args:
        df: Sorted dataframe with segment_id column
        config: Configuration object
        scaler: Fitted StandardScaler

    Returns:
        X_seq: np.array of shape (n_sequences, SEQ_LENGTH, n_features)
        y_seq: np.array of integer fatigue labels
    """
    seq_len = config.SEQ_LENGTH
    step = seq_len - config.SEQ_OVERLAP

    X_sequences = []
    y_labels = []

    for seg_id, segment in df.groupby('segment_id'):
        if len(segment) < seq_len:
            print(f"  Skipping segment {seg_id} ({len(segment)} rows < {seq_len})")
            continue

        X_raw = segment[config.FEATURE_COLUMNS].values
        X_scaled = scaler.transform(X_raw)
        y_raw = segment['fatigue_level'].values

        for i in range(0, len(X_scaled) - seq_len + 1, step):
            X_sequences.append(X_scaled[i:i + seq_len])
            y_labels.append(y_raw[i + seq_len - 1])  # Label = last window

    if len(X_sequences) == 0:
        return np.array([]), np.array([])

    return np.array(X_sequences), np.array(y_labels)


# =============================================================================
# PERSONALIZATION CLASS
# =============================================================================

class UserPersonalization:
    """
    Handles model personalization for individual users.

    Three approaches (tested in order):
      1. Base model + global scaler (baseline)
      2. Base model + user-specific scaler (no retraining)
      3. Fine-tuned model + user-specific scaler (best accuracy)
    """

    def __init__(self, base_model_path=None, global_scaler_path=None):
        """
        Load the base model and global scaler from Phase 1.

        Args:
            base_model_path: Path to trained base model (.h5)
            global_scaler_path: Path to global scaler (.pkl)
        """
        base_model_path = base_model_path or config.BASE_MODEL_PATH
        global_scaler_path = global_scaler_path or config.GLOBAL_SCALER_PATH

        self.base_model_path = base_model_path
        self.base_model = load_model(base_model_path)

        with open(global_scaler_path, 'rb') as f:
            self.global_scaler = pickle.load(f)

        # Load existing user scalers
        try:
            with open(config.USER_SCALERS_PATH, 'rb') as f:
                self.user_scalers = pickle.load(f)
        except FileNotFoundError:
            self.user_scalers = {}

        print("Personalization manager initialized")
        print(f"  Base model: {base_model_path}")
        print(f"  Global scaler: {global_scaler_path}")
        print(f"  Existing user scalers: {len(self.user_scalers)}")

    # -----------------------------------------------------------------
    # Step 1: Create user-specific scaler
    # -----------------------------------------------------------------

    def create_user_scaler(self, df, user_id):
        """
        Fit a StandardScaler on this user's feature data.

        This captures the user's personal baseline so that their
        HR of 90 bpm is normalized relative to THEIR average,
        not the global population average.

        Args:
            df: User's dataframe with feature columns
            user_id: User identifier string

        Returns:
            scaler: Fitted StandardScaler
        """
        print(f"\nCreating scaler for: {user_id}")

        scaler = StandardScaler()
        scaler.fit(df[config.FEATURE_COLUMNS].values)

        # Store in memory
        self.user_scalers[user_id] = scaler

        # Save to pickle (all users)
        with open(config.USER_SCALERS_PATH, 'wb') as f:
            pickle.dump(self.user_scalers, f)

        # Save as JSON for Android deployment
        scaler_json = {
            'user_id': user_id,
            'feature_names': config.FEATURE_COLUMNS,
            'mean': scaler.mean_.tolist(),
            'std': scaler.scale_.tolist(),
        }
        json_path = f'scalers/user_{user_id}_scaler.json'
        with open(json_path, 'w') as f:
            json.dump(scaler_json, f, indent=2)

        print(f"  Mean HR: {scaler.mean_[0]:.1f} bpm")
        print(f"  Std HR:  {scaler.scale_[0]:.1f} bpm")
        print(f"  Saved:   {json_path}")

        return scaler

    # -----------------------------------------------------------------
    # Step 2: Prediction with different normalization
    # -----------------------------------------------------------------

    def predict(self, X_sequences, model=None):
        """
        Run prediction on pre-normalized sequences.

        Args:
            X_sequences: Shape (n_samples, SEQ_LENGTH, n_features), already scaled
            model: Model to use (defaults to base model)

        Returns:
            predictions: Integer class labels
            probabilities: Softmax probabilities per class
        """
        model = model or self.base_model
        probs = model.predict(X_sequences, verbose=0)
        preds = np.argmax(probs, axis=1)
        return preds, probs

    # -----------------------------------------------------------------
    # Step 3: Fine-tuning
    # -----------------------------------------------------------------

    def fine_tune(self, X_train, y_train, X_val, y_val, user_id):
        """
        Fine-tune the base model's classification head for this user.

        Freezes CNN + LSTM layers (general pattern recognition) and
        retrains only the last 3 layers (dense_1, dropout_2, output)
        with a low learning rate on the user's personal data.

        Args:
            X_train, y_train: Training sequences (normalized, one-hot)
            X_val, y_val: Validation sequences
            user_id: User identifier

        Returns:
            model: Fine-tuned model
            history: Training history
        """
        print(f"\nFine-tuning for: {user_id}")
        print(f"  Train: {len(X_train)} sequences")
        print(f"  Val:   {len(X_val)} sequences")

        # Load a fresh copy of the base model
        model = load_model(self.base_model_path)

        # Freeze everything except last 3 layers (dense_1, dropout_2, output)
        for layer in model.layers[:-3]:
            layer.trainable = False

        model.compile(
            optimizer=Adam(learning_rate=config.FINETUNE_LR),
            loss='categorical_crossentropy',
            metrics=['accuracy'],
        )

        trainable = sum(1 for l in model.layers if l.trainable)
        frozen = len(model.layers) - trainable
        print(f"  Frozen: {frozen} layers, Trainable: {trainable} layers")

        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True,
                verbose=1,
            ),
        ]

        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=config.FINETUNE_EPOCHS,
            batch_size=config.FINETUNE_BATCH_SIZE,
            callbacks=callbacks,
            verbose=1,
        )

        # Save
        model_path = f'models/user_{user_id}_model.h5'
        model.save(model_path)
        print(f"  Saved: {model_path}")

        return model, history

    # -----------------------------------------------------------------
    # Step 4: Compare all approaches
    # -----------------------------------------------------------------

    def compare_approaches(self, df, user_id):
        """
        Compare global norm vs user norm vs fine-tuning on the same data.

        Uses an 80/20 split: trains fine-tuning on 80%, evaluates
        all three approaches on the held-out 20%.

        Args:
            df: User's cleaned dataframe (sorted, NaN-free, with segment_id)
            user_id: User identifier

        Returns:
            results: Dict with accuracy for each approach
        """
        print(f"\n{'=' * 60}")
        print(f"COMPARING PERSONALIZATION APPROACHES")
        print(f"User: {user_id}")
        print(f"{'=' * 60}")

        class_names = ["Mild", "Moderate", "High", "Critical"]

        # --- Create user scaler ---
        user_scaler = self.create_user_scaler(df, user_id)

        # --- Build sequences with GLOBAL scaler ---
        X_global, y_seq = create_sequences_from_segments(df, config, self.global_scaler)
        if len(X_global) < 10:
            print(f"\n  Only {len(X_global)} sequences — not enough to compare.")
            return None

        # --- Build sequences with USER scaler ---
        X_user, _ = create_sequences_from_segments(df, config, user_scaler)

        # --- One-hot encode ---
        y_onehot = to_categorical(y_seq, num_classes=config.NUM_CLASSES)

        # --- Split (same indices for fair comparison) ---
        indices = np.arange(len(X_global))

        unique_labels, counts = np.unique(y_seq, return_counts=True)
        can_stratify = len(unique_labels) > 1 and all(c >= 2 for c in counts)

        idx_train, idx_test = train_test_split(
            indices, test_size=0.2, random_state=42,
            stratify=y_seq if can_stratify else None,
        )

        y_true = y_seq[idx_test]

        # --- Approach 1: Base model + global scaler ---
        print("\n--- Approach 1: Base Model + Global Normalization ---")
        preds_global, _ = self.predict(X_global[idx_test])
        acc_global = accuracy_score(y_true, preds_global)
        print(f"  Accuracy: {acc_global:.4f} ({acc_global * 100:.1f}%)")

        # --- Approach 2: Base model + user scaler ---
        print("\n--- Approach 2: Base Model + User Normalization ---")
        preds_user, _ = self.predict(X_user[idx_test])
        acc_user = accuracy_score(y_true, preds_user)
        imp_user = ((acc_user - acc_global) / max(acc_global, 1e-8)) * 100
        print(f"  Accuracy: {acc_user:.4f} ({acc_user * 100:.1f}%)")
        print(f"  vs global: {imp_user:+.1f}%")

        # --- Approach 3: Fine-tuned model + user scaler ---
        acc_ft = None
        if len(idx_train) >= 20:
            print("\n--- Approach 3: Fine-Tuned + User Normalization ---")

            X_ft_train = X_user[idx_train]
            y_ft_train = y_onehot[idx_train]

            # Further split train → train + val for fine-tuning
            X_ft_tr, X_ft_val, y_ft_tr, y_ft_val = train_test_split(
                X_ft_train, y_ft_train, test_size=0.2, random_state=42,
            )

            ft_model, _ = self.fine_tune(X_ft_tr, y_ft_tr, X_ft_val, y_ft_val, user_id)

            preds_ft, _ = self.predict(X_user[idx_test], model=ft_model)
            acc_ft = accuracy_score(y_true, preds_ft)
            imp_ft = ((acc_ft - acc_global) / max(acc_global, 1e-8)) * 100
            print(f"  Accuracy: {acc_ft:.4f} ({acc_ft * 100:.1f}%)")
            print(f"  vs global: {imp_ft:+.1f}%")

            # Classification report for best approach
            best_preds = preds_ft
            print(f"\n  Classification Report (Fine-Tuned):")
            present = sorted(set(y_true) | set(best_preds))
            print(classification_report(
                y_true, best_preds,
                labels=present,
                target_names=[class_names[i] for i in present],
                zero_division=0,
            ))

            # Export fine-tuned model to TFLite
            self._export_user_tflite(user_id)
        else:
            print(f"\n--- Approach 3: Skipped (only {len(idx_train)} train sequences, need 20+) ---")

        # --- Summary ---
        print(f"\n{'=' * 60}")
        print("RESULTS SUMMARY")
        print(f"{'=' * 60}")
        print(f"  Global normalization:    {acc_global:.4f} (baseline)")
        print(f"  User normalization:      {acc_user:.4f} ({imp_user:+.1f}%)")
        if acc_ft is not None:
            print(f"  Fine-tuned + user norm:  {acc_ft:.4f} ({imp_ft:+.1f}%)")

        # Determine best approach
        approaches = {'global': acc_global, 'user_norm': acc_user}
        if acc_ft is not None:
            approaches['fine_tuned'] = acc_ft
        best = max(approaches, key=approaches.get)
        print(f"\n  Best approach: {best} ({approaches[best]:.4f})")

        # Save results
        results = {
            'user_id': user_id,
            'n_windows': len(df),
            'n_sequences': len(X_global),
            'accuracy_global': float(acc_global),
            'accuracy_user_norm': float(acc_user),
            'accuracy_fine_tuned': float(acc_ft) if acc_ft else None,
            'best_approach': best,
        }
        with open(f'results/personalization_{user_id}.json', 'w') as f:
            json.dump(results, f, indent=2)

        return results

    # -----------------------------------------------------------------
    # TFLite export
    # -----------------------------------------------------------------

    def _export_user_tflite(self, user_id):
        """Convert user's fine-tuned model to TFLite for Android."""
        model_path = f'models/user_{user_id}_model.h5'
        tflite_path = f'models/user_{user_id}_model.tflite'

        model = load_model(model_path)
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,
            tf.lite.OpsSet.SELECT_TF_OPS,
        ]
        converter._experimental_lower_tensor_list_ops = False
        tflite_model = converter.convert()

        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)

        size_kb = len(tflite_model) / 1024
        print(f"  TFLite saved: {tflite_path} ({size_kb:.1f} KB)")


# =============================================================================
# MAIN — CLI ENTRY POINT
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Personalize fatigue model for a user")
    parser.add_argument('--user_data', type=str, default='features.csv',
                        help='Path to user\'s features CSV')
    parser.add_argument('--user_id', type=str, default='user_01',
                        help='User identifier')
    parser.add_argument('--base_model', type=str, default=config.BASE_MODEL_PATH,
                        help='Path to base model .h5')
    parser.add_argument('--global_scaler', type=str, default=config.GLOBAL_SCALER_PATH,
                        help='Path to global scaler .pkl')
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("FITGUARD — USER PERSONALIZATION")
    print("=" * 60)

    # Load user data
    df = load_user_data(args.user_data, config)

    if len(df) < config.SEQ_LENGTH:
        print(f"\nERROR: Only {len(df)} labeled rows. Need at least {config.SEQ_LENGTH}.")
        print("Make sure RPE prompts are being collected during the calibration session.")
        return

    # Initialize personalizer
    personalizer = UserPersonalization(args.base_model, args.global_scaler)

    # Compare approaches
    results = personalizer.compare_approaches(df, args.user_id)

    if results is None:
        print("\nNot enough data for personalization comparison.")
        return

    # Final output summary
    print(f"\n{'=' * 60}")
    print("FILES FOR ANDROID DEPLOYMENT")
    print(f"{'=' * 60}")
    print(f"  Model:   models/user_{args.user_id}_model.tflite")
    print(f"  Scaler:  scalers/user_{args.user_id}_scaler.json")
    print(f"\n  Copy both to app/src/main/assets/ on the Android project.")
    print(f"  The app loads the user scaler JSON to normalize features,")
    print(f"  then feeds sequences of {config.SEQ_LENGTH} windows into the TFLite model.")


if __name__ == "__main__":
    main()
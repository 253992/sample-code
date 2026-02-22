"""
PERSONALIZATION WORKFLOW
========================

This script shows how to personalize the model for new users
after the base model has been trained.

Usage:
1. Train base model first: python hybrid_training_complete.py
2. Then use this script to personalize for specific users
"""

import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pickle
import json

# =============================================================================
# PERSONALIZATION CLASS
# =============================================================================

class UserPersonalization:
    """
    Handles personalization for individual users
    """
    
    def __init__(self, base_model_path='models/base_fatigue_model.h5'):
        """
        Initialize personalization manager
        
        Args:
            base_model_path: Path to trained base model
        """
        self.base_model_path = base_model_path
        self.base_model = load_model(base_model_path)
        
        # Load global scaler
        with open('scalers/global_scaler.pkl', 'rb') as f:
            self.global_scaler = pickle.load(f)
        
        # Load user scalers if they exist
        try:
            with open('scalers/user_scalers.pkl', 'rb') as f:
                self.user_scalers = pickle.load(f)
        except FileNotFoundError:
            self.user_scalers = {}
        
        print("✓ Personalization manager initialized")
        print(f"  Base model loaded from: {base_model_path}")
        print(f"  User scalers available: {len(self.user_scalers)}")
    
    def create_user_scaler(self, user_data, user_id):
        """
        Create a user-specific scaler
        
        Args:
            user_data: User's raw feature data (not normalized)
            user_id: User identifier
        
        Returns:
            scaler: Fitted StandardScaler for this user
        """
        print(f"\nCreating scaler for user: {user_id}")
        
        scaler = StandardScaler()
        scaler.fit(user_data)
        
        # Save user scaler
        self.user_scalers[user_id] = scaler
        with open('scalers/user_scalers.pkl', 'wb') as f:
            pickle.dump(self.user_scalers, f)
        
        # Also save as JSON for Android
        scaler_params = {
            'user_id': user_id,
            'mean': scaler.mean_.tolist(),
            'std': scaler.scale_.tolist(),
        }
        with open(f'scalers/user_{user_id}_scaler.json', 'w') as f:
            json.dump(scaler_params, f, indent=2)
        
        print(f"✓ User scaler created and saved")
        print(f"  Mean HR: {scaler.mean_[0]:.2f}")
        print(f"  Std HR: {scaler.scale_[0]:.2f}")
        
        return scaler
    
    def predict_with_global(self, X):
        """
        Predict using base model with global normalization
        
        Args:
            X: Input sequences (already normalized with global scaler)
        
        Returns:
            predictions: Predicted fatigue levels
            probabilities: Prediction probabilities
        """
        probs = self.base_model.predict(X, verbose=0)
        preds = np.argmax(probs, axis=1)
        return preds, probs
    
    def predict_with_user_scaler(self, X_raw, user_id):
        """
        Predict using base model with user-specific normalization
        
        Args:
            X_raw: Raw input data (not normalized)
            user_id: User identifier
        
        Returns:
            predictions: Predicted fatigue levels
            probabilities: Prediction probabilities
        """
        # Get user scaler
        if user_id not in self.user_scalers:
            print(f"Warning: No scaler for user {user_id}, using global")
            scaler = self.global_scaler
        else:
            scaler = self.user_scalers[user_id]
        
        # Normalize with user scaler
        original_shape = X_raw.shape
        X_flat = X_raw.reshape(-1, X_raw.shape[-1])
        X_normalized = scaler.transform(X_flat)
        X_normalized = X_normalized.reshape(original_shape)
        
        # Predict
        probs = self.base_model.predict(X_normalized, verbose=0)
        preds = np.argmax(probs, axis=1)
        
        return preds, probs
    
    def fine_tune_for_user(self, user_X, user_y, user_id, epochs=15):
        """
        Fine-tune base model for specific user
        
        Args:
            user_X: User's feature data (normalized)
            user_y: User's labels (one-hot encoded)
            user_id: User identifier
            epochs: Number of fine-tuning epochs
        
        Returns:
            personalized_model: Fine-tuned model for user
        """
        print(f"\n{'='*60}")
        print(f"FINE-TUNING MODEL FOR USER: {user_id}")
        print(f"{'='*60}")
        
        # Clone base model
        personalized_model = load_model(self.base_model_path)
        
        # Freeze early layers (keep general knowledge)
        for layer in personalized_model.layers[:-2]:
            layer.trainable = False
        
        # Recompile with lower learning rate
        personalized_model.compile(
            optimizer=Adam(learning_rate=0.0001),
            loss="categorical_crossentropy",
            metrics=["accuracy"]
        )
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            user_X, user_y, test_size=0.2, random_state=42
        )
        
        print(f"Training samples: {len(X_train)}")
        print(f"Validation samples: {len(X_val)}")
        
        # Fine-tune
        history = personalized_model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=16,
            verbose=1
        )
        
        # Save personalized model
        model_path = f'models/user_{user_id}_model.h5'
        personalized_model.save(model_path)
        print(f"\n✓ Personalized model saved: {model_path}")
        
        return personalized_model, history
    
    def compare_approaches(self, user_X_raw, user_X_normalized, user_y, user_id):
        """
        Compare different personalization approaches
        
        Args:
            user_X_raw: User's raw data
            user_X_normalized: User's data normalized with global scaler
            user_y: User's labels
            user_id: User identifier
        
        Returns:
            results: Dictionary of comparison results
        """
        print(f"\n{'='*60}")
        print(f"COMPARING PERSONALIZATION APPROACHES")
        print(f"User: {user_id}")
        print(f"{'='*60}")
        
        y_true = np.argmax(user_y, axis=1)
        
        # Approach 1: Global normalization
        print("\n1. Base Model + Global Normalization")
        preds_global, _ = self.predict_with_global(user_X_normalized)
        acc_global = accuracy_score(y_true, preds_global)
        print(f"   Accuracy: {acc_global:.4f} ({acc_global*100:.2f}%)")
        
        # Approach 2: User-specific normalization
        print("\n2. Base Model + User Normalization")
        preds_user, _ = self.predict_with_user_scaler(user_X_raw, user_id)
        acc_user = accuracy_score(y_true, preds_user)
        improvement_1 = (acc_user - acc_global) / acc_global * 100
        print(f"   Accuracy: {acc_user:.4f} ({acc_user*100:.2f}%)")
        print(f"   Improvement: {improvement_1:+.2f}%")
        
        # Approach 3: Fine-tuning (if enough data)
        if len(user_X_normalized) >= 100:
            print("\n3. Fine-Tuned Model + User Normalization")
            
            # Prepare data with user scaler
            user_scaler = self.user_scalers.get(user_id, self.global_scaler)
            original_shape = user_X_raw.shape
            X_flat = user_X_raw.reshape(-1, user_X_raw.shape[-1])
            X_normalized = user_scaler.transform(X_flat)
            X_normalized = X_normalized.reshape(original_shape)
            
            # Fine-tune
            personalized_model, _ = self.fine_tune_for_user(
                X_normalized, user_y, user_id, epochs=10
            )
            
            # Evaluate
            preds_ft = np.argmax(personalized_model.predict(X_normalized, verbose=0), axis=1)
            acc_ft = accuracy_score(y_true, preds_ft)
            improvement_2 = (acc_ft - acc_global) / acc_global * 100
            print(f"   Accuracy: {acc_ft:.4f} ({acc_ft*100:.2f}%)")
            print(f"   Improvement: {improvement_2:+.2f}%")
        else:
            print("\n3. Fine-Tuning: Not enough data (need 100+ samples)")
            acc_ft = None
        
        # Summary
        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")
        print(f"Global normalization:        {acc_global:.4f} (baseline)")
        print(f"User normalization:          {acc_user:.4f} ({improvement_1:+.2f}%)")
        if acc_ft is not None:
            print(f"Fine-tuning + User norm:     {acc_ft:.4f} ({improvement_2:+.2f}%)")
        
        return {
            'global': acc_global,
            'user_norm': acc_user,
            'fine_tuned': acc_ft,
        }


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

def example_personalization():
    """
    Example workflow for personalizing a model for a user
    """
    
    # Load sample data
    print("Loading data...")
    df = pd.read_csv("fitguard_wearable_data.csv")
    
    # Ensure user_id exists
    if 'user_id' not in df.columns:
        df['user_id'] = 'user_' + (df.index // 1000).astype(str)
    
    # Get a sample user
    user_id = df['user_id'].unique()[0]
    user_df = df[df['user_id'] == user_id]
    
    print(f"\nUser: {user_id}")
    print(f"Samples: {len(user_df)}")
    
    # Initialize personalization
    personalizer = UserPersonalization()
    
    # Extract features
    FEATURE_COLUMNS = [
        "heart_rate", "hrv_rmssd", "hrv_sdnn", "ppg_mean",
        "accel_x", "accel_y", "accel_z",
        "gyro_x", "gyro_y", "gyro_z",
        "sleep_debt", "activity_encoded", "rpe"
    ]
    
    # Encode activities if needed
    if 'activity_encoded' not in user_df.columns:
        from sklearn.preprocessing import LabelEncoder
        encoder = LabelEncoder()
        user_df['activity_encoded'] = encoder.fit_transform(user_df['activity_type'])
    
    # Get raw features
    user_X_raw = user_df[FEATURE_COLUMNS].values
    user_y = user_df['fatigue_level'].values
    
    # Create user scaler
    personalizer.create_user_scaler(user_X_raw, user_id)
    
    # For demonstration, create some windowed data
    # (In real app, this would be done by your windowing function)
    # Here we just use a simple approach for demo
    
    # Create dummy sequences for demo (replace with your actual windowing)
    WINDOW_SIZE = 60
    SEQ_LENGTH = 5
    
    # Simple windowing (you'd use your actual windowing function)
    if len(user_X_raw) > WINDOW_SIZE * SEQ_LENGTH:
        # Normalize with global scaler for comparison
        user_X_global_norm = personalizer.global_scaler.transform(user_X_raw)
        
        # Create windows (simplified)
        num_sequences = (len(user_X_global_norm) - WINDOW_SIZE * SEQ_LENGTH) // 10
        sequences_global = []
        sequences_raw = []
        labels = []
        
        for i in range(min(num_sequences, 50)):  # Limit for demo
            start = i * 10
            seq_data_global = []
            seq_data_raw = []
            for j in range(SEQ_LENGTH):
                window_start = start + j * 12
                window_end = window_start + WINDOW_SIZE
                if window_end < len(user_X_global_norm):
                    seq_data_global.append(user_X_global_norm[window_start:window_end])
                    seq_data_raw.append(user_X_raw[window_start:window_end])
            
            if len(seq_data_global) == SEQ_LENGTH:
                sequences_global.append(seq_data_global)
                sequences_raw.append(seq_data_raw)
                labels.append(user_y[start + WINDOW_SIZE * SEQ_LENGTH])
        
        if sequences_global:
            user_X_global = np.array(sequences_global)
            user_X_raw_seq = np.array(sequences_raw)
            user_y_seq = to_categorical(labels, num_classes=4)
            
            print(f"\nCreated {len(user_X_global)} sequences for testing")
            
            # Compare approaches
            results = personalizer.compare_approaches(
                user_X_raw_seq,
                user_X_global,
                user_y_seq,
                user_id
            )
            
            return results
    
    print("\nNot enough data for full demonstration")
    return None


if __name__ == "__main__":
    results = example_personalization()

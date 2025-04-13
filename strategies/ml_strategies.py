"""
Machine learning based trading strategies
"""

import numpy as np
import pandas as pd
from strategies.strategy_base import Strategy
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit
import joblib
import os


class MLStrategy(Strategy):
    def __init__(self, model_type='random_forest', lookback_period=20,
                 prediction_horizon=5, retrain_frequency=50):
        """
        Initialize ML-based trading strategy

        Args:
            model_type (str): Type of ML model ('random_forest', 'gradient_boosting')
            lookback_period (int): Number of periods to use for feature creation
            prediction_horizon (int): Number of periods ahead to predict
            retrain_frequency (int): How often to retrain the model
        """
        super().__init__(
            name=f"ML_{model_type}",
            params={
                "model_type": model_type,
                "lookback_period": lookback_period,
                "prediction_horizon": prediction_horizon,
                "retrain_frequency": retrain_frequency
            }
        )

        self.model = None
        self.feature_columns = None
        self.last_train_idx = 0
        self.min_train_samples = 100

    def _prepare_features(self, data):
        """Prepare features for ML model"""
        lookback = self.params["lookback_period"]
        X = pd.DataFrame(index=data.index)

        # Use existing technical indicators as features
        for col in data.columns:
            if col not in ['open', 'high', 'low', 'close', 'volume']:
                X[col] = data[col]

        # Add lag features
        for lag in range(1, min(lookback + 1, 11)):  # Up to 10 lags
            X[f'close_lag_{lag}'] = data['close'].shift(lag)
            X[f'return_lag_{lag}'] = data['close'].pct_change(lag)

        # Add price difference features
        X['price_diff'] = data['close'] - data['open']

        # Add rolling statistics
        for window in [5, 10, 20]:
            X[f'close_rolling_mean_{window}'] = data['close'].rolling(window=window).mean()
            X[f'close_rolling_std_{window}'] = data['close'].rolling(window=window).std()
            X[f'volume_rolling_mean_{window}'] = data['volume'].rolling(window=window).mean()

        # Drop NaN values
        X = X.dropna()

        return X

    def _prepare_target(self, data, horizon):
        """Prepare target variable for ML model"""
        # Future return as target
        y = data['close'].pct_change(horizon).shift(-horizon)

        # Convert to binary classification
        y_binary = pd.Series(0, index=y.index)
        y_binary[y > 0] = 1  # 1 for positive future returns

        return y_binary

    def _create_model(self):
        """Create ML model based on specified type"""
        model_type = self.params["model_type"]

        if model_type == 'random_forest':
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('model', RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    min_samples_split=10,
                    min_samples_leaf=5,
                    random_state=42
                ))
            ])
        elif model_type == 'gradient_boosting':
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('model', GradientBoostingClassifier(
                    n_estimators=100,
                    max_depth=5,
                    learning_rate=0.1,
                    random_state=42
                ))
            ])
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        return pipeline

    def _train_model(self, X, y):
        """Train ML model"""
        if len(X) < self.min_train_samples:
            self.logger.warning(f"Not enough data for training: {len(X)} < {self.min_train_samples}")
            return False

        # Create time series CV
        tscv = TimeSeriesSplit(n_splits=3)

        # Create and train model
        self.model = self._create_model()
        self.model.fit(X, y)
        self.feature_columns = X.columns

        # Save trained model
        self._save_model()

        self.logger.info(f"Model trained on {len(X)} samples with {len(self.feature_columns)} features")
        return True

    def _save_model(self):
        """Save trained model to disk"""
        try:
            os.makedirs('models', exist_ok=True)
            model_path = f'models/{self.name}_model.pkl'
            joblib.dump(self.model, model_path)
            self.logger.info(f"Model saved to {model_path}")
        except Exception as e:
            self.logger.error(f"Failed to save model: {str(e)}")

    def _load_model(self):
        """Load trained model from disk"""
        try:
            model_path = f'models/{self.name}_model.pkl'
            if os.path.exists(model_path):
                self.model = joblib.load(model_path)
                self.logger.info(f"Model loaded from {model_path}")
                return True
            else:
                self.logger.warning(f"Model file not found: {model_path}")
                return False
        except Exception as e:
            self.logger.error(f"Failed to load model: {str(e)}")
            return False

    def generate_signals(self, data):
        """Generate trading signals using ML predictions"""
        # Get parameters
        lookback = self.params["lookback_period"]
        horizon = self.params["prediction_horizon"]
        retrain_freq = self.params["retrain_frequency"]

        # Prepare features
        X = self._prepare_features(data)

        # Prepare target
        y = self._prepare_target(data, horizon)

        # Initialize signals
        signals = pd.Series(0, index=data.index)

        # Check if we need to train model
        current_idx = len(data)
        if self.model is None or (current_idx - self.last_train_idx) >= retrain_freq:
            # Try to load pre-trained model
            if self.model is None and not self._load_model():
                # Train new model if loading failed
                if self._train_model(X.iloc[:-horizon], y.iloc[:-horizon]):
                    self.last_train_idx = current_idx
                else:
                    self.logger.warning("Could not train or load model, no signals generated")
                    return signals
            elif self.model is not None:
                # Retrain existing model
                if self._train_model(X.iloc[:-horizon], y.iloc[:-horizon]):
                    self.last_train_idx = current_idx

        # Make predictions if model exists
        if self.model is not None and self.feature_columns is not None:
            # Ensure we have all the required features
            missing_features = [f for f in self.feature_columns if f not in X.columns]
            if missing_features:
                self.logger.warning(f"Missing features: {missing_features}")
                return signals

            # Get the most recent data point
            latest_X = X.iloc[-1:][self.feature_columns]

            # Make prediction
            try:
                pred = self.model.predict(latest_X)[0]
                prob = self.model.predict_proba(latest_X)[0]

                # Generate signal based on prediction
                signal_strength = abs(prob[1] - 0.5) * 2  # Scale between 0 and 1

                if pred == 1 and prob[1] > 0.6:  # Strong bullish prediction
                    signals.iloc[-1] = 1
                elif pred == 0 and prob[0] > 0.6:  # Strong bearish prediction
                    signals.iloc[-1] = -1
            except Exception as e:
                self.logger.error(f"Prediction error: {str(e)}")

        return signals


class EnsembleMLStrategy(MLStrategy):
    def __init__(self, models=['random_forest', 'gradient_boosting'],
                 lookback_period=20, prediction_horizon=5, retrain_frequency=50):
        """
        Initialize Ensemble ML strategy that combines multiple ML models

        Args:
            models (list): List of ML models to include in ensemble
            lookback_period (int): Number of periods to use for feature creation
            prediction_horizon (int): Number of periods ahead to predict
            retrain_frequency (int): How often to retrain the model
        """
        super().__init__(
            model_type='ensemble',
            lookback_period=lookback_period,
            prediction_horizon=prediction_horizon,
            retrain_frequency=retrain_frequency
        )

        self.params["models"] = models
        self.ensemble_models = {}

    def _create_ensemble_models(self):
        """Create multiple ML models for ensemble"""
        models = {}

        for model_type in self.params["models"]:
            if model_type == 'random_forest':
                models[model_type] = Pipeline([
                    ('scaler', StandardScaler()),
                    ('model', RandomForestClassifier(
                        n_estimators=100,
                        max_depth=10,
                        random_state=42
                    ))
                ])
            elif model_type == 'gradient_boosting':
                models[model_type] = Pipeline([
                    ('scaler', StandardScaler()),
                    ('model', GradientBoostingClassifier(
                        n_estimators=100,
                        max_depth=5,
                        learning_rate=0.1,
                        random_state=42
                    ))
                ])
            # Add more model types as needed

        return models

    def _train_model(self, X, y):
        """Train ensemble of ML models"""
        if len(X) < self.min_train_samples:
            self.logger.warning(f"Not enough data for training: {len(X)} < {self.min_train_samples}")
            return False

        # Create ensemble models
        self.ensemble_models = self._create_ensemble_models()

        # Train each model
        for model_name, model in self.ensemble_models.items():
            try:
                model.fit(X, y)
                self.logger.info(f"Trained {model_name} model")
            except Exception as e:
                self.logger.error(f"Failed to train {model_name} model: {str(e)}")

        self.feature_columns = X.columns

        # Save trained model
        self._save_model()

        return True

    def _save_model(self):
        """Save trained ensemble models to disk"""
        try:
            os.makedirs('models', exist_ok=True)
            for model_name, model in self.ensemble_models.items():
                model_path = f'models/{self.name}_{model_name}_model.pkl'
                joblib.dump(model, model_path)

            # Also save feature columns
            feature_path = f'models/{self.name}_features.pkl'
            joblib.dump(self.feature_columns, feature_path)

            self.logger.info(f"Ensemble models saved to models/ directory")
        except Exception as e:
            self.logger.error(f"Failed to save ensemble models: {str(e)}")

    def _load_model(self):
        """Load trained ensemble models from disk"""
        try:
            self.ensemble_models = {}
            all_loaded = True

            for model_type in self.params["models"]:
                model_path = f'models/{self.name}_{model_type}_model.pkl'
                if os.path.exists(model_path):
                    self.ensemble_models[model_type] = joblib.load(model_path)
                    self.logger.info(f"Loaded {model_type} model from {model_path}")
                else:
                    self.logger.warning(f"Model file not found: {model_path}")
                    all_loaded = False

            # Load feature columns
            feature_path = f'models/{self.name}_features.pkl'
            if os.path.exists(feature_path):
                self.feature_columns = joblib.load(feature_path)

            return all_loaded and len(self.ensemble_models) > 0
        except Exception as e:
            self.logger.error(f"Failed to load ensemble models: {str(e)}")
            return False

    def generate_signals(self, data):
        """Generate trading signals using ensemble ML predictions"""
        # Get parameters
        lookback = self.params["lookback_period"]
        horizon = self.params["prediction_horizon"]
        retrain_freq = self.params["retrain_frequency"]

        # Prepare features
        X = self._prepare_features(data)

        # Prepare target
        y = self._prepare_target(data, horizon)

        # Initialize signals
        signals = pd.Series(0, index=data.index)

        # Check if we need to train models
        current_idx = len(data)
        if not self.ensemble_models or (current_idx - self.last_train_idx) >= retrain_freq:
            # Try to load pre-trained models
            if not self.ensemble_models and not self._load_model():
                # Train new models if loading failed
                if self._train_model(X.iloc[:-horizon], y.iloc[:-horizon]):
                    self.last_train_idx = current_idx
                else:
                    self.logger.warning("Could not train or load models, no signals generated")
                    return signals
            elif self.ensemble_models:
                # Retrain existing models
                if self._train_model(X.iloc[:-horizon], y.iloc[:-horizon]):
                    self.last_train_idx = current_idx

        # Make predictions if models exist
        if self.ensemble_models and self.feature_columns is not None:
            # Ensure we have all the required features
            missing_features = [f for f in self.feature_columns if f not in X.columns]
            if missing_features:
                self.logger.warning(f"Missing features: {missing_features}")
                return signals

            # Get the most recent data point
            latest_X = X.iloc[-1:][self.feature_columns]

            # Make predictions from all models and average
            predictions = []
            probabilities = []

            for model_name, model in self.ensemble_models.items():
                try:
                    pred = model.predict(latest_X)[0]
                    prob = model.predict_proba(latest_X)[0][1]  # Probability of class 1

                    predictions.append(pred)
                    probabilities.append(prob)
                except Exception as e:
                    self.logger.error(f"Prediction error for {model_name}: {str(e)}")

            if predictions:
                # Average predictions and probabilities
                avg_pred = round(sum(predictions) / len(predictions))
                avg_prob = sum(probabilities) / len(probabilities)

                # Generate signal based on ensemble prediction
                if avg_pred == 1 and avg_prob > 0.6:  # Strong bullish prediction
                    signals.iloc[-1] = 1
                elif avg_pred == 0 and avg_prob < 0.4:  # Strong bearish prediction
                    signals.iloc[-1] = -1

        return signals
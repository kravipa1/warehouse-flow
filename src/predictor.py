"""
Bottleneck Prediction Model

Predicts which stage will become the bottleneck in the next shift/day
based on historical patterns and current conditions.

Uses simple ML (Random Forest / Logistic Regression) - not deep learning.
"""

import pandas as pd
import numpy as np
from typing import Optional, Tuple
from dataclasses import dataclass
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
import joblib
from pathlib import Path


@dataclass
class PredictionResult:
    """Result of bottleneck prediction."""
    predicted_bottleneck: str
    probability: float
    all_probabilities: dict[str, float]
    risk_factors: list[str]
    confidence_level: str  # "high", "medium", "low"


class BottleneckPredictor:
    """
    Predicts future bottlenecks based on operational features.

    Features used:
    - Orders per hour (current and historical)
    - Stage cycle times
    - Worker counts
    - Day of week / hour of day
    - Recent queue buildup trends
    """

    def __init__(self, model_type: str = "random_forest"):
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_names = None
        self.is_trained = False

    def extract_features(
        self,
        event_log: pd.DataFrame,
        window_hours: int = 4
    ) -> pd.DataFrame:
        """
        Extract predictive features from event log.

        Creates rolling window features that capture:
        - Load patterns
        - Stage performance trends
        - Time-based patterns
        """
        df = event_log.copy()
        df["start_time"] = pd.to_datetime(df["start_time"])
        df["hour"] = df["start_time"].dt.floor("h")

        # Aggregate by hour and stage
        hourly = df.groupby(["hour", "stage"]).agg(
            order_count=("order_id", "count"),
            avg_cycle_time=("cycle_time_minutes", "mean"),
            avg_queue_time=("queue_time_minutes", "mean"),
            total_items=("item_count", "sum"),
            unique_workers=("worker_id", "nunique")
        ).reset_index()

        # Pivot to get stage columns
        features_list = []

        for hour in hourly["hour"].unique():
            hour_data = hourly[hourly["hour"] == hour]

            feature_row = {
                "hour": hour,
                "hour_of_day": hour.hour,
                "day_of_week": hour.dayofweek,
                "is_weekend": 1 if hour.dayofweek >= 5 else 0,
                "is_peak_hour": 1 if hour.hour in [9, 10, 11, 14, 15, 16] else 0
            }

            # Per-stage features
            for stage in ["receiving", "picking", "packing", "sorting", "shipping"]:
                stage_data = hour_data[hour_data["stage"] == stage]

                if len(stage_data) > 0:
                    row = stage_data.iloc[0]
                    feature_row[f"{stage}_orders"] = row["order_count"]
                    feature_row[f"{stage}_cycle_time"] = row["avg_cycle_time"]
                    feature_row[f"{stage}_queue_time"] = row["avg_queue_time"]
                    feature_row[f"{stage}_items"] = row["total_items"]
                else:
                    feature_row[f"{stage}_orders"] = 0
                    feature_row[f"{stage}_cycle_time"] = 0
                    feature_row[f"{stage}_queue_time"] = 0
                    feature_row[f"{stage}_items"] = 0

            # Total load
            feature_row["total_orders"] = hour_data["order_count"].sum()
            feature_row["total_items"] = hour_data["total_items"].sum()

            features_list.append(feature_row)

        features_df = pd.DataFrame(features_list)

        # Add rolling features (last N hours)
        for stage in ["receiving", "picking", "packing", "sorting", "shipping"]:
            features_df[f"{stage}_queue_trend"] = features_df[f"{stage}_queue_time"].diff().fillna(0)
            features_df[f"{stage}_cycle_trend"] = features_df[f"{stage}_cycle_time"].diff().fillna(0)

        # Rolling averages
        features_df["orders_rolling_4h"] = features_df["total_orders"].rolling(window=4, min_periods=1).mean()

        return features_df

    def determine_bottleneck_label(
        self,
        event_log: pd.DataFrame,
        hour: pd.Timestamp
    ) -> str:
        """
        Determine which stage was the bottleneck for a given hour.

        Uses a simplified scoring based on queue time.
        """
        df = event_log.copy()
        df["start_time"] = pd.to_datetime(df["start_time"])
        df["hour"] = df["start_time"].dt.floor("h")

        hour_data = df[df["hour"] == hour]

        if len(hour_data) == 0:
            return "none"

        # Score by queue time (primary indicator)
        stage_scores = hour_data.groupby("stage")["queue_time_minutes"].mean()

        if stage_scores.max() < 2:  # No significant bottleneck
            return "balanced"

        return stage_scores.idxmax()

    def prepare_training_data(
        self,
        event_log: pd.DataFrame,
        lookahead_hours: int = 1
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare features and labels for training.

        Features are from time T, labels are bottleneck at time T + lookahead.
        """
        features = self.extract_features(event_log)

        # Create labels (bottleneck in next period)
        labels = []
        for i, row in features.iterrows():
            future_hour = row["hour"] + pd.Timedelta(hours=lookahead_hours)
            label = self.determine_bottleneck_label(event_log, future_hour)
            labels.append(label)

        features["target"] = labels

        # Remove rows where we can't determine future bottleneck
        features = features[features["target"] != "none"]

        # Separate features and target
        feature_cols = [c for c in features.columns if c not in ["hour", "target"]]
        X = features[feature_cols]
        y = features["target"]

        self.feature_names = feature_cols

        return X, y

    def train(
        self,
        event_log: pd.DataFrame,
        lookahead_hours: int = 1,
        test_size: float = 0.2
    ) -> dict:
        """
        Train the prediction model.

        Returns training metrics.
        """
        X, y = self.prepare_training_data(event_log, lookahead_hours)

        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=test_size, random_state=42, stratify=y_encoded
        )

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Train model
        if self.model_type == "random_forest":
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                random_state=42,
                class_weight="balanced"
            )
        else:
            self.model = LogisticRegression(
                max_iter=1000,
                random_state=42,
                class_weight="balanced"
            )

        self.model.fit(X_train_scaled, y_train)
        self.is_trained = True

        # Evaluate
        train_score = self.model.score(X_train_scaled, y_train)
        test_score = self.model.score(X_test_scaled, y_test)

        y_pred = self.model.predict(X_test_scaled)

        # Cross-validation
        cv_scores = cross_val_score(self.model, X_train_scaled, y_train, cv=5)

        return {
            "train_accuracy": train_score,
            "test_accuracy": test_score,
            "cv_mean": cv_scores.mean(),
            "cv_std": cv_scores.std(),
            "classification_report": classification_report(
                y_test, y_pred,
                target_names=self.label_encoder.classes_,
                output_dict=True
            ),
            "feature_importance": self._get_feature_importance()
        }

    def _get_feature_importance(self) -> dict:
        """Get feature importance from the model."""
        if self.model_type == "random_forest":
            importance = self.model.feature_importances_
        else:
            importance = np.abs(self.model.coef_).mean(axis=0)

        return dict(zip(self.feature_names, importance))

    def predict(
        self,
        event_log: pd.DataFrame,
        current_hour: Optional[pd.Timestamp] = None
    ) -> PredictionResult:
        """
        Predict the next bottleneck.

        Args:
            event_log: Recent operational data
            current_hour: Hour to predict from (defaults to latest in data)

        Returns:
            PredictionResult with prediction and confidence
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")

        features = self.extract_features(event_log)

        if current_hour is None:
            current_hour = features["hour"].max()

        # Get features for current hour
        current_features = features[features["hour"] == current_hour]

        if len(current_features) == 0:
            # Use latest available
            current_features = features.iloc[[-1]]

        feature_cols = [c for c in current_features.columns if c not in ["hour", "target"]]
        X = current_features[feature_cols].values

        # Scale and predict
        X_scaled = self.scaler.transform(X)
        prediction = self.model.predict(X_scaled)[0]
        probabilities = self.model.predict_proba(X_scaled)[0]

        # Decode prediction
        predicted_stage = self.label_encoder.inverse_transform([prediction])[0]

        # Get probability for predicted class
        pred_probability = max(probabilities)

        # All probabilities
        all_probs = dict(zip(self.label_encoder.classes_, probabilities))

        # Determine confidence level
        if pred_probability > 0.7:
            confidence_level = "high"
        elif pred_probability > 0.5:
            confidence_level = "medium"
        else:
            confidence_level = "low"

        # Identify risk factors
        risk_factors = self._identify_risk_factors(current_features.iloc[0])

        return PredictionResult(
            predicted_bottleneck=predicted_stage,
            probability=pred_probability,
            all_probabilities=all_probs,
            risk_factors=risk_factors,
            confidence_level=confidence_level
        )

    def _identify_risk_factors(self, features: pd.Series) -> list[str]:
        """Identify current risk factors from features."""
        risks = []

        # High order volume
        if features.get("total_orders", 0) > 30:
            risks.append(f"High order volume ({features['total_orders']:.0f} orders/hour)")

        # Check each stage for warning signs
        for stage in ["picking", "packing", "sorting"]:
            queue_time = features.get(f"{stage}_queue_time", 0)
            if queue_time > 5:
                risks.append(f"{stage.capitalize()} queue building ({queue_time:.1f} min avg)")

            queue_trend = features.get(f"{stage}_queue_trend", 0)
            if queue_trend > 2:
                risks.append(f"{stage.capitalize()} queue time increasing rapidly")

        # Peak hour warning
        if features.get("is_peak_hour", 0) == 1:
            risks.append("Currently in peak demand hours")

        return risks[:5]  # Top 5 risks

    def save(self, path: str):
        """Save trained model to disk."""
        if not self.is_trained:
            raise ValueError("Model not trained. Cannot save.")

        model_data = {
            "model": self.model,
            "scaler": self.scaler,
            "label_encoder": self.label_encoder,
            "feature_names": self.feature_names,
            "model_type": self.model_type
        }
        joblib.dump(model_data, path)

    def load(self, path: str):
        """Load trained model from disk."""
        model_data = joblib.load(path)
        self.model = model_data["model"]
        self.scaler = model_data["scaler"]
        self.label_encoder = model_data["label_encoder"]
        self.feature_names = model_data["feature_names"]
        self.model_type = model_data["model_type"]
        self.is_trained = True


if __name__ == "__main__":
    from data_generator import generate_multi_scenario_dataset
    import pandas as pd

    print("Generating multi-scenario training data...")
    scenarios = generate_multi_scenario_dataset(random_seed=42)

    # Combine all scenarios for training
    training_data = pd.concat(scenarios.values(), ignore_index=True)
    print(f"Total events: {len(training_data)}")

    print("\nTraining bottleneck predictor...")
    predictor = BottleneckPredictor(model_type="random_forest")
    metrics = predictor.train(training_data, lookahead_hours=1)

    print(f"\nModel Performance:")
    print(f"  Train Accuracy: {metrics['train_accuracy']:.2%}")
    print(f"  Test Accuracy: {metrics['test_accuracy']:.2%}")
    print(f"  CV Score: {metrics['cv_mean']:.2%} (+/- {metrics['cv_std']:.2%})")

    print("\nTop 10 Important Features:")
    importance = sorted(
        metrics["feature_importance"].items(),
        key=lambda x: x[1],
        reverse=True
    )[:10]
    for feat, imp in importance:
        print(f"  {feat}: {imp:.4f}")

    print("\nMaking prediction on packing bottleneck scenario...")
    result = predictor.predict(scenarios["packing_bottleneck"])
    print(f"\nPredicted Bottleneck: {result.predicted_bottleneck.upper()}")
    print(f"Confidence: {result.confidence_level} ({result.probability:.1%})")
    print("\nRisk Factors:")
    for risk in result.risk_factors:
        print(f"  • {risk}")

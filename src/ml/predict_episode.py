from pathlib import Path

import joblib
import pandas as pd


MODEL_PATH = Path(
    "models/sentinel2_logistic_regression.joblib"
)

SCHEMA_PATH = Path(
    "models/sentinel2_feature_schema.txt"
)


def load_model():
    """Load trained model and feature schema."""

    model = joblib.load(MODEL_PATH)

    with open(
        SCHEMA_PATH,
        "r",
        encoding="utf-8",
    ) as f:
        features = [
            line.strip()
            for line in f
            if line.strip()
        ]

    return model, features


def predict_episode(
    episode_features: dict,
):
    """
    Predict the class of a single thermal episode.

    Parameters
    ----------
    episode_features : dict
        Dictionary containing all 45 model features.

    Returns
    -------
    dict
        Prediction, confidence and class probabilities.
    """

    model, features = load_model()

    # --------------------------------------------------------
    # Validate features
    # --------------------------------------------------------

    missing = [
        feature
        for feature in features
        if feature not in episode_features
    ]

    if missing:
        raise ValueError(
            f"Missing features: {missing}"
        )

    # --------------------------------------------------------
    # Construct feature vector
    # --------------------------------------------------------

    X = pd.DataFrame(
        [
            {
                feature: episode_features[feature]
                for feature in features
            }
        ]
    )

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    prediction = model.predict(X)[0]

    probabilities = model.predict_proba(X)[0]

    classes = model.classes_

    probability_dict = {
        class_name: float(probability)
        for class_name, probability
        in zip(classes, probabilities)
    }

    confidence = float(
        max(probabilities)
    )

    # --------------------------------------------------------
    # Result
    # --------------------------------------------------------

    return {
        "predicted_class": prediction,
        "confidence": confidence,
        "probabilities": probability_dict,
    }


if __name__ == "__main__":

    DATASET = Path(
        "data/features/final_ml_dataset.parquet"
    )

    print("Loading model...")
    model, features = load_model()

    print(
        f"Loaded model with {len(features)} features."
    )

    print("\nClasses:")
    for class_name in model.classes_:
        print(f"  - {class_name}")

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    df = pd.read_parquet(DATASET)

    # --------------------------------------------------------
    # Select 5 examples from each class
    # --------------------------------------------------------

    samples = []

    for class_name in model.classes_:

        class_df = df[
            df["weak_label"] == class_name
        ]

        # Random but reproducible selection
        selected = class_df.sample(
            n=min(5, len(class_df)),
            random_state=42,
        )

        samples.append(selected)

    test_df = pd.concat(
        samples,
        ignore_index=True,
    )

    # --------------------------------------------------------
    # Run predictions
    # --------------------------------------------------------

    results = []

    for _, episode in test_df.iterrows():

        episode_features = {
            feature: episode[feature]
            for feature in features
        }

        prediction = predict_episode(
            episode_features
        )

        results.append(
            {
                "episode_id": episode["episode_id"],
                "actual": episode["weak_label"],
                "predicted": prediction[
                    "predicted_class"
                ],
                "confidence": prediction[
                    "confidence"
                ],
            }
        )

    results_df = pd.DataFrame(results)

    # --------------------------------------------------------
    # Display results
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("MULTI-CLASS INFERENCE SANITY TEST")
    print("=" * 80)

    print(
        results_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    # --------------------------------------------------------
    # Accuracy
    # --------------------------------------------------------

    accuracy = (
        results_df["actual"]
        == results_df["predicted"]
    ).mean()

    print(
        f"\nSanity-test accuracy: "
        f"{accuracy:.4f}"
    )

    # --------------------------------------------------------
    # Per-class accuracy
    # --------------------------------------------------------

    print("\nPer-class results:")

    for class_name in model.classes_:

        subset = results_df[
            results_df["actual"] == class_name
        ]

        class_accuracy = (
            subset["actual"]
            == subset["predicted"]
        ).mean()

        mean_confidence = (
            subset["confidence"].mean()
        )

        print(
            f"{class_name:30s} "
            f"accuracy={class_accuracy:.4f} "
            f"mean_confidence={mean_confidence:.4f}"
        )
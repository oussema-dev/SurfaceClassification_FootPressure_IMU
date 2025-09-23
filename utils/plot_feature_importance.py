import pandas as pd
import matplotlib.pyplot as plt

def plot_feature_importance(final_model, features):
    # Get feature importance scores
    feature_importance = final_model.feature_importances_

    # Create a DataFrame with feature names and their importance scores
    feature_importance_df = pd.DataFrame(
        {"Feature": features, "Importance": feature_importance}
    )

    # Sort the DataFrame by importance in descending order
    feature_importance_df = feature_importance_df.sort_values(
        by="Importance", ascending=False
    )

    # Select the top 5 features
    top_5_features = feature_importance_df.head(5)

    # Plot the top 5 features
    plt.figure(figsize=(12, 6))
    plt.barh(top_5_features["Feature"][::-1], top_5_features["Importance"][::-1])
    plt.xlabel("Feature Importance")
    plt.title("Top 5 Features Of The XGBoost Model")
    plt.show()
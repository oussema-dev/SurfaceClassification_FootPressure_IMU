import random
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from utils.compute_metrics import compute_metrics
from utils.plot_confusion_matrix import plot_confusion_matrix
from utils.plot_feature_importance import plot_feature_importance

def train_ml_model(df, features, unique_vals):
    y = df["walk_mode"]
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)
    participants = df["participant_id"].unique()
    accuracies = []
    f1_scores = []
    sensitivities = []
    specificities = []
    for i in range(10):
        print ("Iteration", i+1)
        random.seed(i)
        random.shuffle(participants)
        train_percentage = 0.8
        num_train = int(train_percentage * len(participants))
        train_participants = participants[:num_train]
        test_participants = participants[num_train:]
        train_set = df[df["participant_id"].isin(train_participants)]
        test_set = df[df["participant_id"].isin(test_participants)]
        X_train, y_train = train_set[features], train_set["walk_mode"]
        X_test, y_test = test_set[features], test_set["walk_mode"]
        param_grid = {
            "learning_rate": [0.1],
            "max_depth": [3, 5],
            "n_estimators": [100, 150],
        }
        stratified_group_kfold = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=2)
        grid_search = GridSearchCV(
            estimator=XGBClassifier(),
            param_grid=param_grid,
            cv=stratified_group_kfold.split(X_train, y_train, groups=train_set["participant_id"]),
            scoring="accuracy",
            n_jobs=-1,
        )
        grid_search.fit(X_train, y_train)
        best_params = grid_search.best_params_
        final_model = XGBClassifier(**best_params)
        final_model.fit(X_train, y_train)
        y_pred = final_model.predict(X_test)
        acc, f1, sensitivity, specificity = compute_metrics(y_test, y_pred)
        accuracies.append(acc)
        f1_scores.append(f1)
        sensitivities.append(sensitivity)
        specificities.append(specificity)
        if i == 9:
                plot_feature_importance(final_model, features)
                plot_confusion_matrix(y_test, y_pred, unique_vals)
    return accuracies, f1_scores, sensitivities, specificities
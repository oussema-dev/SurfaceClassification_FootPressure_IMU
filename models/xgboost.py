import random
import numpy as np
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score, confusion_matrix
from utils.plot_confusion_matrix import plot_confusion_matrix
from utils.plot_feature_importance import plot_feature_importance

def train_ml_model(df, features):
    y = df["walk_mode"]
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)
    participants = df["participant_id"].unique()
    random.seed(2)
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
    return final_model, y_test, y_pred

def print_results(df, features, unique_vals, model, true, pred):
    y = df["walk_mode"]
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)
    participants = df["participant_id"].unique()
    accuracies = []
    f1_scores = []
    sensitivities = []
    specificities = []
    for i in range(1, 10):
        print ("Iteration", i)
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
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='macro')
        sensitivity = recall_score(y_test, y_pred, average='macro')
        conf_matrix = confusion_matrix(y_test, y_pred)
        specificity_per_class = []
        for i in range(conf_matrix.shape[0]):
            tn = np.sum(conf_matrix) - (np.sum(conf_matrix[i, :]) + np.sum(conf_matrix[:, i]) - conf_matrix[i, i])
            fp = np.sum(conf_matrix[:, i]) - conf_matrix[i, i]
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            specificity_per_class.append(specificity)
        specificity = np.mean(specificity_per_class)
        accuracies.append(accuracy)
        f1_scores.append(f1)
        sensitivities.append(sensitivity)
        specificities.append(specificity)
    print('Mean accuracy:', np.mean(accuracies))
    print('Accuracy standard deviation:', np.std(accuracies))
    print('Mean f1 score:', np.mean(f1_scores))
    print('f1 score standard deviation:', np.std(f1_scores))
    print('Mean sensitivity:', np.mean(sensitivities))
    print('Sensitivity standard deviation:', np.std(sensitivities))
    print('Mean specificity:', np.mean(specificities))
    print('Specificity standard deviation:', np.std(specificities))
    plot_feature_importance(model, features)
    plot_confusion_matrix(true, pred, unique_vals)

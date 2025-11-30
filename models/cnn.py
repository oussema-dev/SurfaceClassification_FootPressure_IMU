import os
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from datetime import datetime
from keras_tuner import RandomSearch
from utils.compute_metrics import compute_metrics
from utils.plot_confusion_matrix import plot_confusion_matrix

# Hybrid Conv1D → TCN → LSTM + MHA (Subject-wise split)
def train_dl(df, grouping_col, unique_vals, i):
    df['walk_mode_enc'] = df['walk_mode'].astype(int)
    # Compute number of classes globally (prevents train/test mismatch)
    class_values = sorted(df['walk_mode_enc'].unique())
    num_classes = len(class_values)
    class_names = [f"class_{c}" for c in class_values]

    # Participant-wise split
    participant_ids = df['participant_id'].unique()
    train_ids, test_ids = train_test_split(participant_ids, test_size=0.2, random_state=i)

    train_df = df[df['participant_id'].isin(train_ids)]
    test_df = df[df['participant_id'].isin(test_ids)]

    # Select input features
    exclude_cols = ['participant_id', 'walk_mode', 'walk_mode_enc', grouping_col, 'task']
    input_cols = [col for col in df.columns if col not in exclude_cols]

    # Scale inputs
    scaler = StandardScaler()
    train_df[input_cols] = scaler.fit_transform(train_df[input_cols])
    test_df[input_cols] = scaler.transform(test_df[input_cols])

    # Sequence reshape
    def reshape(df, group_col=grouping_col):
        grouped = df.groupby(group_col)
        X = np.stack([group[input_cols].values for _, group in grouped])  # (N, T, F)
        y = np.array([group["walk_mode_enc"].iloc[0] for _, group in grouped])
        return X, y

    X_train_full, y_train_full = reshape(train_df)
    X_test, y_test = reshape(test_df)
    y_train_full_cat = to_categorical(y_train_full, num_classes)

    # Class weights
    cw_arr = compute_class_weight("balanced", classes=np.unique(y_train_full), y=y_train_full)
    class_weights = dict(zip(np.unique(y_train_full), cw_arr))

    total = sum(class_weights.values())
    for k in class_weights:
        class_weights[k] = total / (len(class_weights) * class_weights[k])

    # Temporal Convolutional Network Block
    def TCN_block(x, filters, kernel_size, dilation):
        shortcut = x  # residual
        x = layers.Conv1D(filters, kernel_size, dilation_rate=dilation, padding="causal", activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv1D(filters, kernel_size, dilation_rate=dilation, padding="causal", activation="relu")(x)
        x = layers.BatchNormalization()(x)

        # match dimensions for residual
        if shortcut.shape[-1] != x.shape[-1]:
            shortcut = layers.Conv1D(filters, 1, padding="same")(shortcut)

        return layers.Add()([x, shortcut])

    # Build Model (Conv1D → TCN → LSTM → MHA)
    def build_model(hp):
        T, F = X_train_full.shape[1], X_train_full.shape[2]
        inp = layers.Input(shape=(T, F))

        # Conv Block 1
        x = layers.Conv1D(
            filters=hp.Int('filters1', 32, 128, step=32),
            kernel_size=hp.Choice('kernel_size1', [3, 5, 7]),
            activation='relu',
            padding='same'
        )(inp)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling1D(pool_size=2)(x)

        # Conv Block 2
        x = layers.Conv1D(
            filters=hp.Int('filters2', 32, 128, step=32),
            kernel_size=hp.Choice('kernel_size2', [3, 5]),
            activation='relu',
            padding='same'
        )(x)
        x = layers.BatchNormalization()(x)

        # TCN Block (2 layers of dilation)
        x = TCN_block(x, filters=64, kernel_size=3, dilation=1)
        x = TCN_block(x, filters=64, kernel_size=3, dilation=2)

        # LSTM Block
        x = layers.Bidirectional(layers.LSTM(hp.Int("lstm_units", 32, 128, step=32), return_sequences=True))(x)

        # Multi-Head Self-Attention (MHA)
        mha_heads = hp.Choice("mha_heads", [2, 4, 8])
        x = layers.MultiHeadAttention(num_heads=mha_heads, key_dim=32)(x, x)
        x = layers.GlobalAveragePooling1D()(x)

        # Dense Block
        x = layers.Dropout(hp.Float("dropout", 0.2, 0.5))(x)
        x = layers.Dense(hp.Int("dense_units", 32, 128, step=32), activation="relu")(x)

        out = layers.Dense(num_classes, activation="softmax")(x)
        model = models.Model(inp, out)

        lr = hp.Choice("learning_rate", [1e-4, 5e-4, 1e-3])
        model.compile(optimizer=Adam(lr), loss="categorical_crossentropy", metrics=["accuracy"])
        return model

    # Hyperparameter Tuning
    tuner = RandomSearch(build_model, objective='val_accuracy', max_trials=5, directory='cnn_tuner', project_name='honda_project_' + datetime.now().strftime("%Y%m%d-%H%M%S"))
    tuner.search(X_train_full, y_train_full_cat, epochs=10, validation_split=0.2, class_weight=class_weights)

    # Best model
    model = tuner.get_best_models(1)[0]

    # Training with callbacks
    log_dir = os.path.join("logs", datetime.now().strftime("%Y%m%d-%H%M%S"))
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),
        TensorBoard(log_dir=log_dir, histogram_freq=1)
    ]
    model.fit(X_train_full, y_train_full_cat, validation_split=0.2, epochs=50, class_weight=class_weights, callbacks=callbacks, verbose=1)

    # Evaluate
    y_pred = np.argmax(model.predict(X_test), axis=1)
    print("Classification Report:\n", classification_report(y_test, y_pred, target_names=class_names))
    acc, f1, sensitivity, specificity = compute_metrics(y_test, y_pred)

    if i == 10:
        plot_confusion_matrix(y_test, y_pred, unique_vals)

    return acc, f1, sensitivity, specificity

def train_dl_model(df, grouping_col, unique_vals):
    accuracies = []
    f1_scores = []
    sensitivities = []
    specificities = []
    for i in range(10):
        print ("Iteration", i+1)
        acc, f1, sensitivity, specificity = train_dl(df, grouping_col, unique_vals, i+1)
        accuracies.append(acc)
        f1_scores.append(f1)
        sensitivities.append(sensitivity)
        specificities.append(specificity)
    return accuracies, f1_scores, sensitivities, specificities
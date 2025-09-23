import os
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from datetime import datetime
from keras_tuner import RandomSearch
from utils.compute_metrics import compute_metrics
from utils.plot_confusion_matrix import plot_confusion_matrix

def train_dl_model(df, grouping_col, unique_vals):
    le = LabelEncoder()
    df['walk_mode_enc'] = le.fit_transform(df['walk_mode'])

    # Subject-wise split
    participant_ids = df['participant_id'].unique()
    train_ids, test_ids = train_test_split(participant_ids, test_size=0.2, random_state=42)

    train_df = df[df['participant_id'].isin(train_ids)]
    test_df = df[df['participant_id'].isin(test_ids)]

    # Feature columns
    exclude_cols = ['participant_id', 'walk_mode', 'walk_mode_enc', grouping_col]
    input_cols = [col for col in df.columns if col not in exclude_cols]

    # Scaling
    scaler = StandardScaler()
    train_df[input_cols] = scaler.fit_transform(train_df[input_cols])
    test_df[input_cols] = scaler.transform(test_df[input_cols])

    def reshape(df, group_col=grouping_col):
        grouped = df.groupby(group_col)
        X = np.stack([group[input_cols].values for _, group in grouped])
        y = np.array([group['walk_mode_enc'].iloc[0] for _, group in grouped])
        return X, y

    X_train, y_train = reshape(train_df)
    X_test, y_test = reshape(test_df)

    num_classes = len(le.classes_)
    y_train_cat = to_categorical(y_train, num_classes)
    y_test_cat = to_categorical(y_test, num_classes)

    class_weights_array = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
    class_weights = dict(zip(np.unique(y_train), class_weights_array))
    total = sum(class_weights.values())
    for k in class_weights:
        class_weights[k] = total / (len(class_weights) * class_weights[k])

    def build_cnn(hp):
        model = models.Sequential()
        model.add(layers.Conv1D(filters=hp.Int('filters', 32, 128, step=32), kernel_size=hp.Choice('kernel_size', values=[3, 5, 7]), activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])))
        model.add(layers.MaxPooling1D(pool_size=2))
        model.add(layers.Conv1D(filters=hp.Int('filters2', 32, 128, step=32), kernel_size=hp.Choice('kernel_size2', values=[3, 5]), activation='relu'))
        model.add(layers.GlobalAveragePooling1D())
        model.add(layers.Dropout(hp.Float('dropout', 0.2, 0.5)))
        model.add(layers.Dense(hp.Int('dense_units', 32, 128, step=32), activation='relu'))
        model.add(layers.Dense(num_classes, activation='softmax'))
        learning_rate = hp.Choice('learning_rate', values=[1e-4, 5e-4, 1e-3])
        model.compile(optimizer=Adam(learning_rate=learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])
        model.summary()
        return model
    tuner = RandomSearch(build_cnn, objective='val_accuracy', max_trials=5, directory='cnn_tuner', project_name='honda_project' + datetime.now().strftime("%Y%m%d-%H%M%S"))

    tuner.search(X_train, y_train_cat, epochs=10, validation_split=0.2, class_weight=class_weights)

    model = tuner.get_best_models(1)[0]
    log_dir = os.path.join("logs", datetime.now().strftime("%Y%m%d-%H%M%S"))
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),
        TensorBoard(log_dir=log_dir, histogram_freq=1)
    ]
    history = model.fit(X_train, y_train_cat, validation_data=(X_test, y_test_cat), epochs=50, class_weight=class_weights, callbacks=callbacks, verbose=1)

    y_pred = np.argmax(model.predict(X_test), axis=1)
    print("Classification Report:\n", classification_report(y_test, y_pred, target_names=le.classes_))
    labels = np.unique(y_test)
    acc, f1, sensitivity, specificity = compute_metrics(y_test, y_pred, labels)

    print(f"Global Accuracy     : {acc:.4f}")
    print(f"Global F1 Score     : {f1:.4f}")
    print(f"Global Sensitivity  : {sensitivity:.4f}")
    print(f"Global Specificity  : {specificity:.4f}")

    plot_confusion_matrix(y_test, y_pred, unique_vals)
from tensorflow import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau


def build_lstm_model(input_shape, lstm_units=[128, 64, 32], dropout_rate=0.2):
    model = Sequential()

    model.add(LSTM(units=lstm_units[0], return_sequences=True, input_shape=input_shape))
    model.add(Dropout(dropout_rate))

    for units in lstm_units[1:-1]:
        model.add(LSTM(units=units, return_sequences=True))
        model.add(Dropout(dropout_rate))

    model.add(LSTM(units=lstm_units[-1], return_sequences=False))
    model.add(Dropout(dropout_rate))

    model.add(Dense(units=25, activation="relu"))
    model.add(Dense(units=1))  # Saída: preço de fechamento

    model.compile(optimizer="adam", loss="mean_squared_error", metrics=["mae"])

    return model


def get_callbacks(model_path="../models/best_lstm_model.keras"):

    early_stopping = EarlyStopping(
        monitor="val_loss", patience=15, restore_best_weights=True, verbose=1
    )

    model_checkpoint = ModelCheckpoint(
        model_path, monitor="val_loss", save_best_only=True, verbose=1
    )

    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6, verbose=1
    )

    return [early_stopping, model_checkpoint, reduce_lr]


def train_model(
    model, X_train, y_train, X_val, y_val, epochs=100, batch_size=32, callbacks=None
):
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    return history

'''
Number of finished trials: 81
Best trial:
  Value: 0.06268388032913208
  Params: 
    LSTM Neurons: 92
    Dropout Rate: 0.23429094656174487
    learning_rate: 0.00011610230657160765
'''


import optuna
import pandas as pd
from keras.backend import clear_session
from keras.layers import Input, LSTM, Dense, Dropout
from keras.models import Sequential
from keras.optimizers import Adam
from fetch_data import fetch_fear_and_greed_btc
from DataPreprocessor import DataPreprocessor
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler


data = fetch_fear_and_greed_btc()
X_scaler = RobustScaler()
y_scaler = RobustScaler()
preprocessor = DataPreprocessor(X_scaler, y_scaler)

N_TRAIN_EXAMPLES = 3000
N_VALID_EXAMPLES = 1000
BATCHSIZE = 64
VALIDATION_SPLIT = 0.25
CLASSES = 10
EPOCHS = 100

def objective(trial):
    # Clear clutter from previous Keras session graphs.
    clear_session()

    X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, _, _ = preprocessor.preprocess_data(data)

    # Print shapes for debugging
    print(f"X_train_scaled shape: {X_train_scaled.shape}")
    print(f"X_test_scaled shape: {X_test_scaled.shape}")

    X_train_scaled = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1])) 
    X_test_scaled = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))

    timesteps = X_train_scaled.shape[1] 
    features = X_train_scaled.shape[2]
    
    model = Sequential()
    model.add(Input(shape=(timesteps, features)))
    model.add(LSTM(units=trial.suggest_int('LSTM Neurons_0', 10, 100), return_sequences=True))
    model.add(Dropout(trial.suggest_float('Dropout Rate_0', .0001, .50)))
    # model.add(LSTM(units=trial.suggest_int('LSTM Neurons_1', 10, 1000), return_sequences=False))
    # model.add(Dropout(trial.suggest_float('Dropout Rate_1 ', .0001, .50)))
    model.add(Dense(trial.suggest_int('Dense Neurons', 1, 50), activation='relu'))
    model.add(Dense(1))  # No activation for regression

    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-1, log=True)
    model.compile(
        loss='mean_squared_error',
        optimizer=Adam(learning_rate=learning_rate),
        metrics=['mean_absolute_error']
    )
    
    model.fit(
        X_train_scaled,
        y_train_scaled,
        # validation_data=(X_test_scaled, y_test_scaled),
        shuffle=False,
        epochs=EPOCHS,
        batch_size=BATCHSIZE,
        validation_split=VALIDATION_SPLIT,
        verbose=False,
    )

    # Evaluate the model accuracy on the validation set.
    score = model.evaluate(X_test_scaled, y_test_scaled, verbose=0)
    return score[1]


if __name__ == "__main__":
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=10, timeout=100_000)

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
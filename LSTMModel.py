from datetime import datetime
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM, Input, Dropout
from tensorflow.keras.optimizers import Adam
from plotting_utils import *
from fetch_data import fetch_fear_and_greed_btc
from generate_signals import generate_signal
from backtester_utils import *
from DataPreprocessor import DataPreprocessor
from ModelEvaluator import ModelEvaluator


class LSTMModel:
    def __init__(self, model_path=None, data_path=None, lags=5, test_size=.25, learning_rate = 0.001, epochs=50, batch_size=32, validation_split=0.2, plot=True):
        self.model = None
        self.model_path = model_path
        self.history = None
        self.data_path = data_path
        self.lag_features = ['value', 'Close'] # change these if you want to calculate lags on different feature columns
        self.target_col = 'Close' # change this if you want to target a different variable than Close
        self.X_scaler = RobustScaler()
        self.y_scaler = RobustScaler()
        self.preprocessor = DataPreprocessor(self.X_scaler, self.y_scaler, self.lag_features, lags, self.target_col, test_size)
        self.current_timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        self.learning_rate = learning_rate 
        self.loss = 'mean_squared_error' # change this if you're not going to solve for a regression target
        self.metrics = ['mean_absolute_error']  # change this if you're not going to solve for a regression target
        self.epochs = epochs
        self.batch_size = batch_size
        self.validation_split = validation_split

        self.plot = plot # set plot to false when instantiating if you dont want the backtest graph

        if model_path:
            self.load_saved_model(model_path)
    
    def load_saved_model(self, model_path):
        self.model = load_model(model_path)
        print(f'Model loaded from {model_path}')

    # Loads Data from a User-fed CSV Path, if CSV passed
    def load_data(self):
        if self.data_path is None:
            print("No data path preloaded. Downloading Fear and Greed and BTC data...")
            self.data = fetch_fear_and_greed_btc()
        else:
            print("Data path preloaded. saving csv to dataframe...")
            self.data = pd.read_csv(self.data_path, parse_dates=True, index_col='timestamp') 

    def preprocess_data(self):
        (
            self.X_train_scaled,
            self.X_test_scaled,
            self.y_train_scaled,
            self.y_test_scaled,
            self.X_test,
            self.y_test
        ) = self.preprocessor.preprocess_data(self.data)

    def reshape_for_lstm(self):
        # Reshape from (samples, features) to (samples, 1, features)
        self.X_train_scaled = self.X_train_scaled.reshape((self.X_train_scaled.shape[0], 1, self.X_train_scaled.shape[1])) 
        self.X_test_scaled = self.X_test_scaled.reshape((self.X_test_scaled.shape[0], 1, self.X_test_scaled.shape[1])) 

    def build_model_lstm(self):
        self.reshape_for_lstm()
        timesteps = self.X_train_scaled.shape[1] 
        features = self.X_train_scaled.shape[2] 

        model = Sequential()
        model.add(Input(shape=(timesteps, features)))
        model.add(LSTM(150, return_sequences=False))
        model.add(Dropout(0.50)) # Dropout Regularization
        model.add(Dense(20, activation='relu'))
        model.add(Dense(1))  # No activation for regression
        model.compile(optimizer=Adam(learning_rate=self.learning_rate), loss=self.loss, metrics=self.metrics)
        model.summary()
        self.model = model
        # save_and_visualize_model(self.model)

    def train_model(self):
        self.history = self.model.fit(
            self.X_train_scaled, 
            self.y_train_scaled, 
            epochs = self.epochs, 
            batch_size = self.batch_size, 
            validation_split = self.validation_split, 
            verbose=1
        )

    def plot_training_history(self):
        plot_loss_training_history(self.history)
        plot_mae_training_history(self.history)
    
    def evaluate_model(self):
        self.evaluator = ModelEvaluator(self.model, self.X_test, self.y_test, self.X_test_scaled, self.y_test_scaled, self.y_scaler)
        self.evaluator.evaluate_model()
        self.evaluator.atr_to_data()
    
    def predict_model(self):
        self.predictions_inversed = self.evaluator.predict_model()

    def save_model(self):  
        self.model_path = f'{self.current_timestamp}_LSTM_model_epochs_{self.epochs}.keras'
        self.model.save(self.model_path)
        print("Model saved successfully.")

    def generate_model_signals(self):
        self.X_test = generate_signal(self.X_test, self.predictions_inversed)
        # print(self.X_test)

    def backtest_signals(self):
        run_backtest(data=self.X_test, plot=self.plot)

    def run_and_train(self):
        self.load_data()
        self.preprocess_data()
        self.build_model_lstm()
        self.train_model()
        self.plot_training_history()
        self.evaluate_model()
        self.predict_model()
        self.save_model()
        self.generate_model_signals()
        self.backtest_signals()

    def run_with_pretrained(self):
        self.load_data()
        self.preprocess_data()
        self.reshape_for_lstm()
        self.evaluate_model()
        self.predict_model()
        self.generate_model_signals()
        self.backtest_signals()


model = LSTMModel(test_size=0.25, 
                  learning_rate=0.001, 
                  epochs=50, 
                  batch_size=32, 
                  validation_split=0.25, 
                  plot=True)

model.run_and_train()


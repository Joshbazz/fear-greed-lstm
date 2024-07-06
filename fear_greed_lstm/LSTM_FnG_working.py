import datetime
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Input
from tensorflow.keras.optimizers import Adam
from plotting_utils import *
from fetch_data import fetch_fear_and_greed_btc
from generate_signals import generate_signal
from backtester_live import *


class LSTMModel:
    def __init__(self, data_path=None, lags=5, test_size=.25, learning_rate = 0.001, epochs=50, batch_size=32, validation_split=0.2):
        self.data_path = data_path
        self.lags = lags
        self.lag_features = ['value', 'Close'] # change these if you want to calculate lags on different feature columns
        self.target_col = 'Close' # change this if you want to target a different variable than Close
        self.test_size = test_size 
        self.learning_rate = learning_rate 
        self.loss = 'mean_squared_error' # change this if you're not going to solve for a regression target
        self.metrics = ['mean_absolute_error']  # change this if you're not going to solve for a regression target
        self.epochs = epochs
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.X_scaler = StandardScaler()
        self.y_scaler = StandardScaler()

    
    # Loads Data from a User-fed CSV Path, if CSV passed
    def load_and_preprocess_data(self):
        if self.data_path is None:
            print("No data path preloaded. Downloading Fear and Greed and BTC data...")
            self.data = fetch_fear_and_greed_btc()
        else:
            print("Data path preloaded. saving csv to dataframe...")
            self.data = pd.read_csv(self.data_path, parse_dates=True, index_col='timestamp') 


    def create_lagged_features(self, df): 
        for feature in self.lag_features:
            for lag in range(1, self.lags + 1):
                df[f'{feature}_lag_{lag}'] = df[feature].shift(lag)
        df['target'] = df[self.target_col].shift(-1)
        df.dropna(inplace=True)
        self.lagged_df = df


    def normalize_data(self, X_train, X_test, y_train, y_test):
        # scale the features for train and test
        self.X_scaler.fit(X_train)
        self.X_train_scaled = self.X_scaler.transform(X_train)

        self.X_scaler.fit(X_test)
        self.X_test_scaled = self.X_scaler.transform(X_test)

        # scale the targets for train and test
        y_train = y_train.values.reshape(-1, 1)
        self.y_scaler.fit(y_train)
        self.y_train_scaled = self.y_scaler.transform(y_train)
        
        y_test = y_test.values.reshape(-1, 1)
        self.y_scaler.fit(y_test)
        self.y_test_scaled = self.y_scaler.transform(y_test)

    
    def split_train_test_fear_greed_btc(self):
        self.create_lagged_features(self.data)
        X = self.lagged_df.drop(columns=['target', 'value_classification'])
        y = self.lagged_df['target']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size, shuffle=False)
        # Save X_test for generate_signal() later
        self.X_test = X_test
        self.normalize_data(X_train, X_test, y_train, y_test)


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
        model.add(LSTM(10, return_sequences=False))
        model.add(Dense(5, activation='relu'))
        model.add(Dense(3, activation='relu'))
        model.add(Dense(1))  # No activation for regression
        model.compile(optimizer=Adam(learning_rate=self.learning_rate), loss=self.loss, metrics=self.metrics)
        model.summary()
        self.model = model
        save_and_visualize_model(self.model)


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
        plot_mse_training_history(self.history)
        
    
    def evaluate_predict_model(self):
        loss, mae = self.model.evaluate(self.X_test_scaled, self.y_test_scaled, verbose=2)
        print(f'Test Loss: {loss:.4f}')
        print(f'Test MAE: {mae * 100:.2f}%')
        self.predictions = self.model.predict(self.X_test_scaled)
        self.predictions_inversed = self.y_scaler.inverse_transform(self.predictions).flatten()
        self.y_test_inversed = self.y_scaler.inverse_transform(self.y_test_scaled).flatten()


    def save_model(self):
        current_timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        self.model_path = f'models/{current_timestamp}_LSTM_model_epochs_{self.epochs}.keras'
        self.model.save(self.model_path)
        print("Model saved successfully.")

    def generate_model_signals(self):
        self.X_test = generate_signal(self.X_test, self.predictions_inversed)
        print(self.X_test)


    def backtest_signals(self):
        run_backtest(data=self.X_test)


    def run(self):
        self.load_and_preprocess_data()
        self.split_train_test_fear_greed_btc()
        self.build_model_lstm()
        self.train_model()
        self.plot_training_history()
        self.evaluate_predict_model()
        self.save_model()
        self.generate_model_signals()
        self.backtest_signals()


# Example Run
model = LSTMModel(test_size=0.25, learning_rate=0.001, batch_size=64, validation_split=0.25)
model.run()

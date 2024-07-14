import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

class DataPreprocessor:
    def __init__(self, X_scaler=RobustScaler(), y_scaler=RobustScaler(), lag_features=['value', 'Close'], lags=5, target_col='Close', test_size=.25, window_size=5):
        self.lag_features = lag_features
        self.lags = lags
        self.window_size = window_size
        self.target_col = target_col
        self.test_size = test_size
        self.X_scaler = X_scaler
        self.y_scaler = y_scaler

    def create_lagged_features(self, df):
        for feature in self.lag_features:
            for lag in range(1, self.lags + 1):
                df[f'{feature}_lag_{lag}'] = df[feature].shift(lag)
        df['target'] = df[self.target_col]
        df.dropna(inplace=True)
        return df

    def convert_to_window_format(self, df):
        X, y, dates = [], [], []
        for i in range(len(df) - self.window_size):
            window = df.iloc[i:i+self.window_size]
            X.append(window.drop(columns=['target', 'Close', 'Adj Close']).values)
            # Append the Close price of the last day in the window as the target
            y.append(window.iloc[-1]['Close'])
            dates.append(window.index[-1]) # store the date of the last row in the window
        self.X, self.y = np.array(X), np.array(y)
        self.dates = np.array(dates)

        return self.X, self.y, self.dates

    def normalize_data(self, X_train, X_test, y_train, y_test):
        # Reshape X_train and X_test to fit_transform
        X_train_reshaped = X_train.reshape(-1, X_train.shape[-1])
        X_test_reshaped = X_test.reshape(-1, X_test.shape[-1])

        X_train_scaled = self.X_scaler.fit_transform(X_train_reshaped)
        X_test_scaled = self.X_scaler.transform(X_test_reshaped)
        
        # Reshape back to original shape
        X_train_scaled = X_train_scaled.reshape(X_train.shape)
        X_test_scaled = X_test_scaled.reshape(X_test.shape)

        y_train = y_train.reshape(-1, 1)
        y_train_scaled = self.y_scaler.fit_transform(y_train)
        y_test = y_test.reshape(-1, 1)
        y_test_scaled = self.y_scaler.transform(y_test)

        return X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled

    def split_train_test(self, data):
        lagged_df = self.create_lagged_features(data)
        lagged_df = lagged_df.drop(columns=['value_classification'])

        self.X, self.y, self.dates = self.convert_to_window_format(lagged_df)

        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=self.test_size, shuffle=False)

        X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled = self.normalize_data(
            X_train, X_test, y_train, y_test
        )
        self.dates_train = self.dates[:len(y_train_scaled)]
        self.dates_test = self.dates[len(y_train_scaled):]
        self.features_test_df = lagged_df[-len(self.dates_test)-1:] #NOTE this could cause issues

        return X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, X_test, y_test, self.dates_train, self.dates_test, self.features_test_df

    def preprocess_data(self, data):
        return self.split_train_test(data)

    def inverse_transform_y(self, y_scaled):
        return self.y_scaler.inverse_transform(y_scaled)

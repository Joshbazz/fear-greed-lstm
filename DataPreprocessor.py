import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

class DataPreprocessor:
    def __init__(self, X_scaler, y_scaler, lag_features=['value', 'Close'], lags=5, target_col='Close', test_size=.25):
        self.lag_features = lag_features
        self.lags = lags
        self.target_col = target_col
        self.test_size = test_size
        self.X_scaler = X_scaler
        self.y_scaler = y_scaler

    def create_lagged_features(self, df):
        for feature in self.lag_features:
            for lag in range(1, self.lags + 1):
                df[f'{feature}_lag_{lag}'] = df[feature].shift(lag)
        df['target'] = df[self.target_col].shift(-1)
        df.dropna(inplace=True)
        return df

    def normalize_data(self, X_train, X_test, y_train, y_test):
        X_train_scaled = self.X_scaler.fit_transform(X_train)
        X_test_scaled = self.X_scaler.transform(X_test)
        
        y_train = y_train.values.reshape(-1, 1)
        y_train_scaled = self.y_scaler.fit_transform(y_train)
        y_test = y_test.values.reshape(-1, 1)
        y_test_scaled = self.y_scaler.transform(y_test)

        return X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled

    def split_train_test(self, data):
        lagged_df = self.create_lagged_features(data)
        X = lagged_df.drop(columns=['target', 'value_classification'])
        y = lagged_df['target']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size, shuffle=False)
        
        X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled = self.normalize_data(
            X_train, X_test, y_train, y_test
        )
        
        return X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, X_test, y_test

    def preprocess_data(self, data):
        return self.split_train_test(data)

    def inverse_transform_y(self, y_scaled):
        return self.y_scaler.inverse_transform(y_scaled)
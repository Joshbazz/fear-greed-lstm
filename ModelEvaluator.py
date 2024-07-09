from generate_signals import generate_signal
from plotting_utils import plot_predicted_actual, plot_residuals
from tensorflow.keras.models import load_model

# this needs to link up with DataPreprocessor to get train and test vals
class ModelEvaluator:
    def __init__(self, model, X_test, y_test, X_test_scaled, y_test_scaled, y_scaler, model_path=None):
        self.model = model
        self.model_path = model_path
        self.X_test = X_test
        self.y_test = y_test
        self.X_test_scaled = X_test_scaled
        self.y_test_scaled = y_test_scaled
        self.y_scaler = y_scaler
        self.predictions = None
        self.predictions_inversed = None
        self.y_test_inversed = None

        if not self.model and model_path:
            self.load_saved_model(model_path)    
    
    def load_saved_model(self, model_path):
        self.model = load_model(model_path)
        print(f'Model loaded from {model_path}')

    def evaluate_model(self):
        loss, mae = self.model.evaluate(self.X_test_scaled, self.y_test_scaled, verbose=2)
        error_in_dollars = self.y_test.mean() * mae
        print(f'Test Loss: {loss:.4f}')
        print(f'Test MAE: {mae:.2f}')
        print(f'MAE in dollars: +/- ${error_in_dollars:.2f}')

    def atr_to_data(self, window=30):
        self.X_test['ATR'] = self.calculate_atr()
        atr_total_test = self.X_test['ATR'].mean()
        atr_last_window = self.X_test['ATR'].iloc[-window:].mean()
        print(f"ATR for all test observations: ${atr_total_test:.2f}")
        print(f"ATR for last {window} observations: ${atr_last_window:.2f}")

    def calculate_atr(self, window=14):
        high_low = self.X_test['High'] - self.X_test['Low']
        high_close_prev = abs(self.X_test['High'] - self.X_test['Close'].shift(1))
        low_close_prev = abs(self.X_test['Low'] - self.X_test['Close'].shift(1))

        tr = high_low.to_frame(name='HL')
        tr['HC_prev'] = high_close_prev
        tr['LC_prev'] = low_close_prev

        true_range = tr.max(axis=1)

        atr = true_range.rolling(window=window, min_periods=1).mean()

        return atr
        
    def predict_model(self):
        self.predictions = self.model.predict(self.X_test_scaled)
        self.predictions_inversed = self.y_scaler.inverse_transform(self.predictions).flatten()
        self.y_test_inversed = self.y_scaler.inverse_transform(self.y_test_scaled).flatten()
        plot_predicted_actual(self.y_test_inversed, self.predictions_inversed)
        plot_residuals(self.y_test_inversed, self.predictions_inversed)

        return self.predictions_inversed

    def generate_model_signals(self):
        self.X_test = generate_signal(self.X_test_scaled, self.predictions_inversed)
        print(self.X_test)

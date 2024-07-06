import pandas as pd
import os
from keras.models import load_model
from sklearn.preprocessing import RobustScaler
import datetime

def generate_signal(test_features, predictions, model_path=None):

    # put your predictions vector back into the test features dataframe
    test_features['predictions'] = predictions

####################################################################################
#------------------------CREATE YOUR STRATEGY HERE---------------------------------#
####################################################################################

    # Initialize an empty list to store signals
    signals = []

    # Iterate through each row of the DataFrame
    for i in range(len(test_features)):
        close_lag_1 = test_features['Close_lag_1'].iloc[i]
        prediction = test_features['predictions'].iloc[i]
        
        # Define your buy and sell conditions here (modular and editable)
        if close_lag_1 < prediction:
            signal = 1  # Buy signal
        else:
            signal = -1  # Sell signal
        
        signals.append(signal)

    # Add the signals list as a new column 'signal' in the DataFrame
    test_features['Signal'] = signals

    # Get the current timestamp and format it
    current_timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

    # Define the full path for the new CSV file
    csv_path = os.path.join(f'{current_timestamp}_new_data_with_positions.csv')

    # Save new_data with positions
    test_features.to_csv(csv_path, index=True)

    return test_features


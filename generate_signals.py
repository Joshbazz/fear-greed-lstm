import pandas as pd
import os
import numpy as np
# from keras.models import load_model
from sklearn.preprocessing import RobustScaler
import datetime


def generate_signal(test_features, predictions, dates_test, model_path=None):

    # put your predictions vector back into the test features dataframe
    dates_test_reshaped = dates_test.reshape(-1, 1)
    # predictions = np.ravel(predictions)
    combined_array = np.concatenate((dates_test_reshaped, predictions), axis=1)
    print(combined_array[:10])

    df_combined = pd.DataFrame(combined_array, columns=['Date', 'Predicted_Close'])
    df_combined.set_index('Date', inplace=True)

    result_df = pd.concat([test_features, df_combined], axis=1)

####################################################################################
#------------------------CREATE YOUR STRATEGY HERE---------------------------------#
####################################################################################


    # Initialize an empty list to store signals
    signals = []

    # Iterate through each row of the DataFrame
    for i in range(len(result_df) - 1):
        open = result_df['Open'].iloc[i]
        prediction = result_df['Predicted_Close'].iloc[i]
        
        # Define your buy and sell conditions here (modular and editable)
        if open < prediction:
            signal = 1  # Buy signal
        else:
            signal = -1  # Sell signal
        
        signals.append(signal)

    # Handle the last element if necessary
    if len(signals) < len(test_features):
        signals.append(None)

    # Add the signals list as a new column 'signal' in the DataFrame
    test_features['Signal'] = signals

    # Get the current timestamp and format it
    current_timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

    # Define the full path for the new CSV file
    csv_path = os.path.join(f'{current_timestamp}_new_data_with_positions.csv')

    # Save new_data with positions
    test_features.to_csv(csv_path, index=True)

    return test_features


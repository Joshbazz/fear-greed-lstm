import requests
import pandas as pd
import yfinance as yf


def fetch_fear_and_greed_btc():
    # Define the API endpoint and parameters for Fear and Greed Index
    fng_api_url = "https://api.alternative.me/fng/"
    fng_params = {
        'limit': 0,  # Get all available data
        'format': 'json'
    }

    # Make the GET request to the Fear and Greed Index API
    response = requests.get(fng_api_url, params=fng_params)

    # Check if the request was successful
    if response.status_code == 200:
        # Parse the JSON response
        fng_data = response.json()
        # Convert the data to a Pandas DataFrame
        fng_df = pd.DataFrame(fng_data['data'])
        # Ensure the timestamp column is of numeric type before converting to datetime
        fng_df['timestamp'] = pd.to_numeric(fng_df['timestamp'], errors='coerce')
        # Convert the timestamp column to datetime
        fng_df['timestamp'] = pd.to_datetime(fng_df['timestamp'], unit='s')
        # Drop the time_until_update column
        fng_df.drop(columns=['time_until_update'], inplace=True)
        # Set the timestamp as the index
        fng_df.set_index('timestamp', inplace=True)
        # Sort the DataFrame by the index (timestamp) in ascending order
        fng_df.sort_index(inplace=True)
        # Save the Fear and Greed Index DataFrame to a CSV file with timestamp as index and column name 'timestamp'
        fng_df.to_csv('fear_and_greed_index.csv', index=True, index_label='timestamp')
        print("Fear and Greed Index data has been saved to 'fear_and_greed_index.csv'.")
    else:
        print(f"Failed to fetch Fear and Greed Index data. Status code: {response.status_code}")

    # Fetch daily Bitcoin prices using Yahoo Finance
    btc_data = yf.download('BTC-USD', start=fng_df.index.min().strftime('%Y-%m-%d'), end=fng_df.index.max().strftime('%Y-%m-%d'))

    # Concatenate Fear and Greed Index DataFrame with Bitcoin DataFrame based on date index
    combined_df = pd.concat([fng_df, btc_data], axis=1, join='inner')

    # Save the combined DataFrame to a CSV file
    combined_df.to_csv('fear_greed_btc_combined.csv', index=True, index_label='timestamp')
    print("Combined data has been saved to 'fear_greed_btc_combined.csv'.")

    return combined_df
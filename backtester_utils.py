import pandas as pd
from datetime import datetime
from backtesting import Backtest, Strategy
    
class SignalStrategy(Strategy):
    def init(self):
        self.signal = self.data.Signal

    def next(self):
        current_signal = self.data.Signal[-1]
        current_date = self.data.index[-1]
        # print(f"Date: {current_date}, Current position size: {self.position.size}, Signal: {current_signal}, Position: {self.position.is_long}")
        
        if current_signal == 1:
            # print("Executing BUY order")
            self.buy(size=1)
        elif current_signal == -1 and self.position.is_long:
            # print("Attempting to SELL entire position")
            try:
                self.position.close()  # This closes the entire position
                # print("SELL order executed - entire position closed")
            except Exception as e:
                print(f"Error executing SELL order: {e}")
        elif current_signal == 0:
            # print("No trade executed")
            pass
    
        
        # print(f"Current position size: {self.position.size}")


def run_backtest(data_path=None, data=None, plot=True, cash=1_000_000, commission=0.002, trade_on_close=True):
    if data_path:
        # Load and preprocess the data from the specified path
        dataframe = pd.read_csv(data_path, index_col='Date', parse_dates=True)
        dataframe = dataframe.sort_index()
        dataframe = dataframe.dropna()
        dataframe = dataframe.drop_duplicates()
        dataframe.columns = [column.capitalize() for column in dataframe.columns]
    elif data is not None:
        # Use self.data if called from LSTMModel instance
        dataframe = data  # Assuming `self.data` is defined in LSTMModel
    
    # Initialize and run the backtest
    bt = Backtest(dataframe, SignalStrategy, cash=cash, commission=commission, trade_on_close=trade_on_close)
    stats = bt.run()

    # Print the statistics and plot the backtest results
    print(stats)

    current_time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    stats_output_file = f'backtest_stats_{current_time}.txt'

    # Save the statistics to a text file if stats_output_file is provided
    with open(stats_output_file, 'w') as f:
        f.write(str(stats))

    if plot == True:
        bt.plot()
    else:
        pass
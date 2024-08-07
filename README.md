# Fear and Greed LSTM Deep Learning Trading Algorithm

[![CCDS Project Template](https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter)](https://cookiecutter-data-science.drivendata.org/) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Joshbazz/fear-greed-lstm/blob/master/colab.ipynb)

This project implements an LSTM Deep Learning Network trained on Fear and Greed Index data for Predicting Closing Bitcoin Prices.

You can view the detailed write-up about the code [here](https://joshbazzano.substack.com/p/c48aaf2c-ff9b-4d1a-9857-e8b027353a3e).

## Run the Code in Google Colab

If you prefer running the code without downloading the repository or if you're a non-technical user, you can run the project directly in Google Colab. Click the badge below to open the notebook in your browser:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Joshbazz/fear-greed-lstm/blob/master/colab.ipynb)

Simply navigate to the top bar, and under Runtime, click on "Run All" (see below):

![Run All in Colab](references/image.png)

Note: Initial downloads may be required when running in Colab.

## Project Structure

Both the Python and Jupyter Notebook implementations are held in the main directory. To run the project:
1. Navigate to the `LSTMModel.py` file in your terminal.
2. Execute the file to perform:
   - Data gathering
   - Preprocessing
   - Model training
   - Training Visualizations
   - Model saving
   - Model evaluation
   - Evaluation Visualizations
   - Signal generation
   - Backtesting the strategy on generated signals

## Example:

There is an example run of a whole backtest, optimized and non-optimized, contained with the backtests folder

## Installation

To install the required packages and dependencies, follow these steps:

1. **Clone the Repository**:
   
   ```bash
   git clone https://github.com/Joshbazz/fear-greed-lstm.git
   cd fear-greed-lstm

3. **Install Make with Conda if not Already Installed (optional but recommended)**
   ```bash
   conda install -c conda-forge make

2. **Create a Virtual Environment with Make (optional but recommended)**
   ```makefile
   make create_environment

3. **Activate the Environment before Downloading Requirements**
   ```bash
   conda activate fear-greed-lstm

4. **Install Dependencies**
   ```makefile
   make requirements

**NOTE:** 
Due to issues with graphviz, in `LSTMModel.py`, the (`save_and_visualize_model(self.model_path)`) is commented out. If you successfully get graphviz installed, you can uncomment.

You'll need to locally install Graphviz and/or Make in order to run the `make` commands and create the model visualization. To download Make for Windows, open up Powershell and run: `winget install ezwinports.make` 

There's an issue where using graphviz on VScode run from the Anaconda Platform is creating issues. Make sure you are running VScode explicitly from your own separate download. VScode can be downloaded [here](https://code.visualstudio.com)

Links for downloading Make (Windows) are [here](https://gnuwin32.sourceforge.net/packages/make.htm), and downloads for Graphviz are included [here](https://graphviz.org).





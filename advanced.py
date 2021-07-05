import subprocess
import sys

# try:
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mplcyberpunk

# Getting the Data in string format from NodeJS
# NodeJS passed 2 args ['./test.py', data] where sys.argv[1] represents DATA
data = sys.argv[1]

# Creating a Pandas DataFrame for the supplied data
my_data_new = pd.DataFrame([x.split(',') for x in data.split('\n')])

# Rearranging and Cleaning the Data
my_data_new.columns = my_data_new.iloc[0]
my_data_new = my_data_new.drop(columns = ['dividend_amount', 'split_coefficient\r'])
my_data_new.drop(0, axis = 0, inplace = True)
my_data_new.drop(101, axis = 0, inplace = True)
my_data_new = my_data_new.iloc[::-1]        # reverse all rows
my_data_new.reset_index(drop = True, inplace = True)

# Converting the Data type of each data item like close, high, low etc from "Object" to "numeric"
for i in range(0, len(my_data_new.columns)):
    my_data_new.iloc[:,i] = pd.to_numeric(my_data_new.iloc[:,i], errors='ignore')

# Important Functions
# 1) this returns the mean absolute error b/w the predicted and the true value
def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# 2) this function plots the moving average and the Required Plot
# accepts a pandas data frame named series => Required Plot
# accepts window size for rolling mean
# accepts name for the png figure to be save
def plot_moving_average(series, window, my_name):

    # creating a DataFrame for storing the rolling mean's
    rolling_mean_prediction = series.rolling(window=window).mean()

    # return the rolling mean DataFrame for the Current Dynamic(close, high, low etc)
    # so as to complete the prediction process
    return rolling_mean_prediction

def exponential_smoothing(series, alpha):
    result = [series[0]] # first value is same as series
    for n in range(1, len(series)):
        result.append(alpha * series[n] + (1 - alpha) * result[n-1])
    return result

def plot_exponential_smoothing(series, alpha, my_name):
    expo_prediction = exponential_smoothing(series, alpha)
    return expo_prediction

def gen_my_plot(prediction, original, my_name, my_technique_label):
    plt.figure(figsize=(20, 10))

    # Plotting the Prediction and Original on the same Plot
    # Plot the prediction
    plt.plot(my_data_new.timestamp, prediction, marker = 'o', label = my_technique_label, color = 'green')

    # # plot the original data on the same plot
    plt.plot(original, label='Actual values', marker = 'D', color = 'red')

    plt.xticks(np.arange(0, 100, step=10))  # Set label locations.
    plt.tick_params(labelsize=12)
    plt.legend(loc='best')
    plt.grid(True)
    plt.xlabel('TIMESTAMP', fontsize = 15)
    if(my_name == 'volume'):
        plt.ylabel('VOLUME (in Units)')
    else:
        lab = 'PRICES'
        if(sys.argv[5] == 'BSE'):
            lab = lab + " (in Rs.)"
        else:
            lab = lab + " (in $)"
        plt.ylabel(lab, fontsize = 15)
    plt.legend()
    mplcyberpunk.add_glow_effects()

    file_name = "/tmp/fig_advanced_prediction_" + my_name + '.png'
    plt.savefig(file_name)

# 3) Prediction function for the Market Dynamic
# We pass last(x2,y2) and 2nd-last mean(x1, y1) and predict x2 + 1
# though we can use this for and point but it will increase error chance for values
def my_prediction(x1, y1, x2, y2, X):
    slope = (y2 - y1) / (x2 - x1)
    y = y1 + slope * (X - x1)
    return (y, slope)

# 4) Market Mood Predicting Function
# uses Slope Value of the last 2 rolling mean values to predict the Up / Down Nature of the Prediction Plot
def mood(slope):
    if(slope >= 0):
        return "Up"
    else:
        return "Down"

# defining number of Rows and Columns into the DataFrame
r,c = my_data_new.shape

# result_prediction variable to store final result
result_prediction = ""

# ===============================   Generating Prediction ===========================
window = int(sys.argv[3])
pleasePlot = sys.argv[2].split()
plot_technique = sys.argv[4]
plt.style.use("cyberpunk")
for dynamic in pleasePlot:
    if(dynamic == 'open'):
        # Plot the Original Data
        plt.figure(figsize=(20, 10))
        plt.plot(my_data_new.timestamp, my_data_new.open, marker = 'd', color = 'red')
        plt.xticks(np.arange(0, 100, step=10))  # Set label locations.
        plt.tick_params(labelsize=12)
        plt.legend(loc='best')
        plt.grid(True)
        plt.xlabel('TIMESTAMP', fontsize = 15)

        lab = 'PRICES'
        if(sys.argv[5] == 'BSE'):
            lab = lab + " (in Rs.)"
        else:
            lab = lab + " (in $)"
        plt.ylabel(lab, fontsize = 15)

        plt.legend()
        mplcyberpunk.add_glow_effects()
        file_name = "/tmp/fig_advanced_opening.png"
        plt.savefig(file_name)

        # Calculating the Prediction Value using MOVING AVERAGE and Exponential Smoothing
        my_technique_label = ""
        if(plot_technique == 'moving_average'):
            prediction = plot_moving_average(my_data_new.open, window, "opening")
            y2 = prediction.iloc[-1]
            x2 = r-1
            y1 = prediction.iloc[-2]
            x1 = r-2
            my_technique_label = "Moving Average"

        else:
            alpha = 2 / (window + 1)
            prediction = plot_exponential_smoothing(my_data_new.open, alpha, "opening")
            y2 = prediction[-1]
            x2 = r-1
            y1 = prediction[-2]
            x1 = r-2
            my_technique_label = "Exponential Smoothing"

        gen_my_plot(prediction, my_data_new.open, "opening", my_technique_label)
        prediction, slope = my_prediction(x1, y1, x2, y2, x2+1)

        # Adding the Results Obtained i.e. Predicted Value and Mood to "result_prediction"
        # using "%" as a separator
        result_prediction += str(prediction) + "%" + mood(slope) + "%"

    elif(dynamic == 'close'):
        # Plot the Original Data
        plt.figure(figsize=(20, 10))
        plt.plot(my_data_new.timestamp, my_data_new.close, marker = 'd', color = 'red')
        plt.xticks(np.arange(0, 100, step=10))  # Set label locations.
        plt.tick_params(labelsize=12)
        plt.legend(loc='best')
        plt.grid(True)
        plt.xlabel('TIMESTAMP', fontsize = 15)

        lab = 'PRICES'
        if(sys.argv[5] == 'BSE'):
            lab = lab + " (in Rs.)"
        else:
            lab = lab + " (in $)"
        plt.ylabel(lab, fontsize = 15)

        plt.legend()
        mplcyberpunk.add_glow_effects()
        file_name = "/tmp/fig_advanced_closing.png"
        plt.savefig(file_name)

        # Calculating the Prediction Value using MOVING AVERAGE and Exponential Smoothing
        my_technique_label = ""
        if(plot_technique == 'moving_average'):
            prediction = plot_moving_average(my_data_new.close, window, "closing")
            y2 = prediction.iloc[-1]
            x2 = r-1
            y1 = prediction.iloc[-2]
            x1 = r-2
            my_technique_label = "Moving Average"

        else:
            alpha = 2 / (window + 1)
            prediction = plot_exponential_smoothing(my_data_new.close, alpha, "closing")
            y2 = prediction[-1]
            x2 = r-1
            y1 = prediction[-2]
            x1 = r-2
            my_technique_label = "Exponential Smoothing"

        gen_my_plot(prediction, my_data_new.close, "closing", my_technique_label)
        prediction, slope = my_prediction(x1, y1, x2, y2, x2+1)

        # Adding the Results Obtained i.e. Predicted Value and Mood to "result_prediction"
        # using "%" as a separator
        result_prediction += str(prediction) + "%" + mood(slope) + "%"


    elif(dynamic == 'high'):
        # Plot the Original Data
        plt.figure(figsize=(20, 10))
        plt.plot(my_data_new.timestamp, my_data_new.high, marker = 'd', color = 'red')
        plt.xticks(np.arange(0, 100, step=10))  # Set label locations.
        plt.tick_params(labelsize=12)
        plt.legend(loc='best')
        plt.grid(True)
        plt.xlabel('TIMESTAMP', fontsize = 15)

        lab = 'PRICES'
        if(sys.argv[5] == 'BSE'):
            lab = lab + " (in Rs.)"
        else:
            lab = lab + " (in $)"
        plt.ylabel(lab, fontsize = 15)

        plt.legend()
        mplcyberpunk.add_glow_effects()
        file_name = "/tmp/fig_advanced_high.png"
        plt.savefig(file_name)

        # Calculating the Prediction Value using MOVING AVERAGE and Exponential Smoothing
        my_technique_label = ""
        if(plot_technique == 'moving_average'):
            prediction = plot_moving_average(my_data_new.high, window, "high")
            y2 = prediction.iloc[-1]
            x2 = r-1
            y1 = prediction.iloc[-2]
            x1 = r-2
            my_technique_label = "Moving Average"

        else:
            alpha = 2 / (window + 1)
            prediction = plot_exponential_smoothing(my_data_new.high, alpha, "high")
            y2 = prediction[-1]
            x2 = r-1
            y1 = prediction[-2]
            x1 = r-2
            my_technique_label = "Exponential Smoothing"

        gen_my_plot(prediction, my_data_new.high, "high", my_technique_label)
        prediction, slope = my_prediction(x1, y1, x2, y2, x2+1)

        # Adding the Results Obtained i.e. Predicted Value and Mood to "result_prediction"
        # using "%" as a separator
        result_prediction += str(prediction) + "%" + mood(slope) + "%"

    elif(dynamic == 'low'):
        # Plot the Original Data
        plt.figure(figsize=(20, 10))
        plt.plot(my_data_new.timestamp, my_data_new.low, marker = 'd', color = 'red')
        plt.xticks(np.arange(0, 100, step=10))  # Set label locations.
        plt.tick_params(labelsize=12)
        plt.legend(loc='best')
        plt.grid(True)
        plt.xlabel('TIMESTAMP', fontsize = 15)

        lab = 'PRICES'
        if(sys.argv[5] == 'BSE'):
            lab = lab + " (in Rs.)"
        else:
            lab = lab + " (in $)"
        plt.ylabel(lab, fontsize = 15)

        plt.legend()
        mplcyberpunk.add_glow_effects()
        file_name = "/tmp/fig_advanced_low.png"
        plt.savefig(file_name)

        # Calculating the Prediction Value using MOVING AVERAGE and Exponential Smoothing
        my_technique_label = ""
        if(plot_technique == 'moving_average'):
            prediction = plot_moving_average(my_data_new.low, window, "low")
            y2 = prediction.iloc[-1]
            x2 = r-1
            y1 = prediction.iloc[-2]
            x1 = r-2
            my_technique_label = "Moving Average"

        else:
            alpha = 2 / (window + 1)
            prediction = plot_exponential_smoothing(my_data_new.low, alpha, "low")
            y2 = prediction[-1]
            x2 = r-1
            y1 = prediction[-2]
            x1 = r-2
            my_technique_label = "Exponential Smoothing"

        gen_my_plot(prediction, my_data_new.low, "low", my_technique_label)
        prediction, slope = my_prediction(x1, y1, x2, y2, x2+1)

        # Adding the Results Obtained i.e. Predicted Value and Mood to "result_prediction"
        # using "%" as a separator
        result_prediction += str(prediction) + "%" + mood(slope) + "%"

    elif(dynamic == 'adjusted_close'):
        # Plot the Original Data
        plt.figure(figsize=(20, 10))
        plt.plot(my_data_new.timestamp, my_data_new.adjusted_close, marker = 'd', color = 'red')
        plt.xticks(np.arange(0, 100, step=10))  # Set label locations.
        plt.tick_params(labelsize=12)
        plt.legend(loc='best')
        plt.grid(True)
        plt.xlabel('TIMESTAMP', fontsize = 15)

        lab = 'PRICES'
        if(sys.argv[5] == 'BSE'):
            lab = lab + " (in Rs.)"
        else:
            lab = lab + " (in $)"
        plt.ylabel(lab, fontsize = 15)

        plt.legend()
        mplcyberpunk.add_glow_effects()
        file_name = "/tmp/fig_advanced_adjusted_close.png"
        plt.savefig(file_name)

        # Calculating the Prediction Value using MOVING AVERAGE and Exponential Smoothing
        my_technique_label = ""
        if(plot_technique == 'moving_average'):
            prediction = plot_moving_average(my_data_new.adjusted_close, window, "adjusted_close")
            y2 = prediction.iloc[-1]
            x2 = r-1
            y1 = prediction.iloc[-2]
            x1 = r-2
            my_technique_label = "Moving Average"

        else:
            alpha = 2 / (window + 1)
            prediction = plot_exponential_smoothing(my_data_new.adjusted_close, alpha, "adjusted_close")
            y2 = prediction[-1]
            x2 = r-1
            y1 = prediction[-2]
            x1 = r-2
            my_technique_label = "Exponential Smoothing"

        gen_my_plot(prediction, my_data_new.adjusted_close, "adjusted_close", my_technique_label)
        prediction, slope = my_prediction(x1, y1, x2, y2, x2+1)

        # Adding the Results Obtained i.e. Predicted Value and Mood to "result_prediction"
        # using "%" as a separator
        result_prediction += str(prediction) + "%" + mood(slope) + "%"

    elif(dynamic == 'volume'):
        # Plot the Original Data
        plt.figure(figsize=(20, 10))
        plt.plot(my_data_new.timestamp, my_data_new.volume, marker = 'd', color = 'red')
        plt.xticks(np.arange(0, 100, step=10))  # Set label locations.
        plt.tick_params(labelsize=12)
        plt.legend(loc='best')
        plt.grid(True)
        plt.xlabel('TIMESTAMP', fontsize = 15)

        lab = 'VOLUME (in Units)'
        # if(sys.argv[5] == 'BSE'):
        #     lab = lab + " (in Rs.)"
        # else:
        #     lab = lab + " (in $)"
        plt.ylabel(lab, fontsize = 15)

        plt.legend()
        mplcyberpunk.add_glow_effects()
        file_name = "/tmp/fig_advanced_volume.png"
        plt.savefig(file_name)

        # Calculating the Prediction Value using MOVING AVERAGE and Exponential Smoothing
        my_technique_label = ""
        if(plot_technique == 'moving_average'):
            prediction = plot_moving_average(my_data_new.volume, window, "volume")
            y2 = prediction.iloc[-1]
            x2 = r-1
            y1 = prediction.iloc[-2]
            x1 = r-2
            my_technique_label = "Moving Average"

        else:
            alpha = 2 / (window + 1)
            prediction = plot_exponential_smoothing(my_data_new.volume, alpha, "volume")
            y2 = prediction[-1]
            x2 = r-1
            y1 = prediction[-2]
            x1 = r-2
            my_technique_label = "Exponential Smoothing"

        gen_my_plot(prediction, my_data_new.volume, "volume", my_technique_label)
        prediction, slope = my_prediction(x1, y1, x2, y2, x2+1)

        # Adding the Results Obtained i.e. Predicted Value and Mood to "result_prediction"
        # using "%" as a separator
        result_prediction += str(prediction) + "%" + mood(slope) + "%"

    else:
        print("No Market Dynamic Found to Plot")


# returning the results
print(result_prediction)

sys.stdout.flush

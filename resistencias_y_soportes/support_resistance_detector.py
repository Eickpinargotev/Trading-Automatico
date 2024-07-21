import numpy as np
import pandas as pd
import pandas_ta as ta

from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import mplfinance as mpf
import scipy
import math

def animacion_resistencias(data: pd.DataFrame, 
                           pdf_x: np.array, 
                           range_y: np.array,
                           lookback: int, 
                           cycle_length=200, 
                           interval=25):
    # Use a dark background style for the plot
    plt.style.use('dark_background')
    
    # Create a figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, gridspec_kw={'width_ratios': [3, 1]}, figsize=(15, 6))

    # Configure initial plot limits and styles for ax1
    ax1.tick_params(axis='x', rotation=45)
    ax1.set_xlim(data.index[0], data.index[cycle_length])
    ax1.set_ylim(data['close'].iloc[0:cycle_length].min(), data['close'].iloc[0:cycle_length].max())

    # Calculate the maximum and minimum values for pdf_x
    pdf_max = np.max(np.concatenate([n for n in pdf_x if n is not None]))
    pdf_min = np.min(np.concatenate([n for n in pdf_x if n is not None]))
    ax2.set_xlim(pdf_max, pdf_min)
    ax2.invert_yaxis()  # Invert the y-axis for ax2

    # Create empty line objects for the animation
    line1, = ax1.plot([], [], 'w-', label='Price')
    line2, = ax2.plot([], [], 'w-', label='Density')
    line3 = ax2.scatter([], [], color='r', label='Peaks', s=100)
    line4 = ax1.scatter([], [], color='y', label='Valleys', s=5)

    def animate(i):
        start = i + lookback

        # Update the price line (line1) with new data up to the current frame
        line1.set_data(data.index[:start], data['close'].iloc[:start])

        # Update the density line (line2) with new data up to the current frame
        line2.set_data(pdf_x[start], range_y[start])

        # Plot peaks on the density chart
        pos = peaks[start]  # Positions of the peaks
        line3.set_offsets(np.column_stack((pdf_x[start][pos], range_y[start][pos])))  # Plot the peaks

        # Plot levels on the price chart
        expanded_dates = []
        expanded_levels = []
        for date, level_array in zip(data[lookback:start].index, data[lookback:start]['levels']):
            expanded_dates.extend([date] * len(level_array))
            expanded_levels.extend(level_array)
        line4.set_offsets(np.column_stack((expanded_dates, expanded_levels)))  # Plot the valleys

        # Adjust the plot limits for ax1 and ax2 based on the current frame
        if start >= cycle_length:
            ax1.set_xlim(data.index[start-cycle_length], data.index[start])
            ax1.set_ylim(data['close'].iloc[start-cycle_length:start].min(), data['close'].iloc[start-cycle_length:start].max())
            # Adjust x-axis limits for ax2 to show only one cycle
            ax2.set_xlim(pdf_x[start].max() + 1, pdf_x[start].min())
        else:
            ax1.set_xlim(data.index[0], data.index[cycle_length])
            ax1.set_ylim(data['close'].iloc[0:cycle_length].min(), data['close'].iloc[0:cycle_length].max())        
            # Adjust x-axis limits for ax2 to show only one cycle
            ax2.set_xlim(pdf_x[start].max() + 1, pdf_x[start].min())
        return line1, line2, line3

    # Create the animation
    ani = FuncAnimation(fig, animate, frames=(len(data) - lookback), interval=interval)

    # Adjust layout and display the plot
    plt.tight_layout()
    plt.show()

def find_levels( 
        price: np.array, 
        atr: float, # Log closing price, and log atr 
        first_w: float = 0.1, 
        atr_mult: float = 3.0, 
        prom_thresh: float = 0.1):

    # Setup weights
    last_w = 1.0
    w_step = (last_w - first_w) / len(price)
    weights = first_w + np.arange(len(price)) * w_step
    weights[weights < 0] = 0.0

    # Get kernel of price. 
    kernal = scipy.stats.gaussian_kde(price, bw_method=atr*atr_mult, weights=weights)

    # Construct market profile
    min_v = np.min(price)
    max_v = np.max(price)
    step = (max_v - min_v) / 200 #200
    price_range = np.arange(min_v, max_v, step)
    pdf = kernal(price_range) # Market profile

    # Find significant peaks in the market profile
    pdf_max = np.max(pdf)
    prom_min = pdf_max * prom_thresh

    peaks, props = scipy.signal.find_peaks(pdf, prominence=prom_min)
    levels = [] 
    for peak in peaks:
        levels.append(np.exp(price_range[peak]))

    return levels, peaks, props, price_range, pdf, weights

def support_resistance_levels(
        data: pd.DataFrame, lookback: int, 
        first_w: float = 0.01, atr_mult:float=3.0, prom_thresh:float =0.25
):

    # Get log average true range, 
    atr = ta.atr(np.log(data['high']), np.log(data['low']), np.log(data['close']), lookback)

    all_levels = [None] * len(data)
    pdf_all = [None] * len(data)
    range_all = [None] * len(data)
    peaks_all = [None] * len(data)

    for i in range(lookback, len(data)):
        i_start  = i - lookback
        vals = np.log(data.iloc[i_start+1: i+1]['close'].to_numpy())
        levels, peaks, props, price_range, pdf, weights= find_levels(vals, atr.iloc[i], first_w, atr_mult, prom_thresh)
        all_levels[i] = levels
        pdf_all[i] = pdf
        range_all[i] = np.exp(price_range)
        peaks_all[i] = peaks

        
    return all_levels, pdf_all, range_all,peaks_all

if __name__ == '__main__':

    # Load the data
    data = pd.read_csv('BTCUSDT86400.csv')
    data['date'] = data['date'].astype('datetime64[s]')
    data = data.set_index('date')

    # Create a test DataFrame
    new_df = data.iloc[0:2000].copy()

    # Calculate support and resistance levels, probability density function, ranges, and peaks
    (levels, pdf, ranges, peaks) = support_resistance_levels(new_df, lookback=299, first_w=1.0, atr_mult=3.0)
    new_df.loc[:, 'levels'] = np.array(levels, dtype=object)

    # Plot the data with the animation function
    animacion_resistencias(new_df, pdf, ranges, lookback=299, cycle_length=300, interval=25)


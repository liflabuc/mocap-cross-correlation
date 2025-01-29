# Motion Capture (MOCAP) Processing Script

# Written by Marisol Correa, 2024, under the supervision of Esteban Hurtado

# This script is designed to process motion capture data files in CSV format from various experimental conditions.
# It computes cross-correlations between time-series data to analyze synchronized movements

# Requirements: Python 3.10

import os
import scipy.stats
import statistics
import math 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.interpolate import interp1d
from scipy.signal import butter, filtfilt
from scipy.stats import norm
import colorsys


# Example commands to run this script from the terminal with different options
# These commands demonstrate how to execute the script with a specified folder and arguments like sampling frequency and interpolation.
    # py -3.10 "G:\My Drive\2024\ICESI\PROYECTO\Chile\Mocap_processing.py" "G:\My Drive\2024\ICESI\PROYECTO\Chile\Datos\Datos" -fs 180   

# Interpolation is optional and it is a useful technique to create smoother and more aesthetically pleasing plots. By applying spline 
# interpolation, you can achieve a more refined visual representation of your data. 

def load_files(condition):
    "Load CSV files from a subfolder (experimental condition)."
    # Join the base folder path with the subfolder name
    condition_folder = os.path.join(folder_path, condition)
    # List all CSV files in the condition folder
    files = [file for file in os.listdir(condition_folder) if file.endswith('.csv')]
    condition_data = {}
    # Read each CSV file into a pandas DataFrame and store it in a dictionary
    for file in files:
        file_path = os.path.join(condition_folder, file)
        series_name = file.split('.')[0]
        condition_data[series_name] = pd.read_csv(file_path, delimiter='\t', header=None)  # Specify delimiter
    return condition_data

def generate_pastel_colors(n):
    # Generate pastel colors by converting HLS values to RGB, ensuring a lightness of 0.7 and saturation of 0.6 for softer colors
    return [colorsys.hls_to_rgb(hue, 0.7, 0.6) for hue in np.linspace(0, 1, n, endpoint=False)]


if __name__ == "__main__":
    # Parse command line arguments for flexibility in specifying input data and analysis parameters
    import argparse
    parser = argparse.ArgumentParser(
        prog = 'Motion cross-correlation',
        description = \
            'Reads two time series from each CSV file (2 columns, as many rows as samples) '
            'in each of several subfolders. Each subfolder contains data for a different '
            'experimental condition. Generates one aggregated cross-correlation curve '
            'with confidence intervals for each subfolder.',
            epilog='')
    parser.add_argument('folder', type=str,
                        help='Data folder that contains subfolders to process.' )
    parser.add_argument('-fs', type=float,
                        help='Sampling frequency for input data.')
    parser.add_argument('-fc', type=float, default=10,
                        help='Low-pass filter cutoff frequency in Hz.')
    parser.add_argument('-min-lag', type=float, default=-1.5,
                        help="Lag time lower limit (inclusive) in seconds.")
    parser.add_argument('-max-lag', type=float, default=1.6,
                        help="Lag time upper limit (exclusive) in seconds.")
    parser.add_argument('-lag-step', type=float, default=0.1,
                        help="Step between lag values in seconds.")
    parser.add_argument('-interpolate', action='store_true', help='Enable spline interpolation for smoothing the plot.')
    args = parser.parse_args()

    # Set parameters based on command-line arguments
    folder_path = args.folder # Path to input data folder
    fs = args.fs  # Sampling frequency in Hz
    fc = args.fc  # Cutoff frequency in Hz
    start_shift = args.min_lag 
    end_shift = args.max_lag   # Adjusted to include 1.5
    shift_step = args.lag_step    

    # Get list of subfolders in the specified folder path
    subfolders = next(os.walk(folder_path))[1]
    num_subfolders = len(subfolders) 
    # Generate pastel colors for plotting each subfolder's data distinctly
    colors = generate_pastel_colors(num_subfolders)
    color_index = 0
    
    plt.figure() 

    # Create a figure and a subplot for plotting 
    # Global (aggregated) plots
    global_fig, global_ax = plt.subplots()

    # Process each subfolder to analyze and visualize motion capture data
    for subfolder in subfolders:
        
        # Lists to store correlation and shift results
        total_correlations = []
        total_lengths = []
        
        # Load motion capture data files for the current condition
        cond = load_files(subfolder)
        files_cond = list(cond.keys())

        # Design a low-pass filter to preprocess the data
        b, a = butter(N=2, Wn=(2 * fc) / fs, btype='low', analog=False)

        for file in files_cond:
            subject_A = cond[file][0] # Extract column 1 as subject A's data
            subject_B = cond[file][1] # Extract column 2 as subject B's data

            # Preprocess data: compute velocity (difference between consecutive samples) and apply low-pass filter
            subject_A = subject_A.diff().dropna()
            subject_B = subject_B.diff().dropna()

            subject_A = filtfilt(b, a, subject_A)
            subject_B = filtfilt(b, a, subject_B)

            # Initialize lists to store correlation results for each shift
            correlations = []
            shifts = []
            lengths = []

            # Compute cross-correlation over a range of time lags
            for shift in np.arange(start_shift, end_shift, shift_step):
                # Apply the time shift
                shift_indices = abs(int(shift * fs))
  
                # Displacing A
                if shift < 0:
                    deviation_A = subject_A[shift_indices:]
                    deviation_B = subject_B[:len(deviation_A)]

                elif shift > 0:
                    deviation_B = subject_B[shift_indices:]
                    deviation_A = subject_A[:len(deviation_B)]

                else:
                    deviation_B = subject_B
                    deviation_A = subject_A
                
                # Signal centering
                deviation_A -= np.mean(deviation_A)
                deviation_B -= np.mean(deviation_B)
                
                # Calculate Pearson correlation coefficient for each pair of centered signals
                correlation, _ = scipy.stats.pearsonr(deviation_A, deviation_B)

                # Store correlation and length of the data used
                correlations.append(correlation) 
                lengths.append(len(deviation_A))
                shifts.append(shift)

            # Aggregate results for all files within the subfolder
            total_correlations.append(correlations)
            total_lengths.append(lengths)


        # Transpose the list of lists to align data for averaging
        transposed = list(zip(*total_correlations))

        # Calculate averages of correlations for each lag time
        averages = [statistics.mean(position) for position in transposed]
       
        # Apply inverse hyperbolic tangent function to each correlation value
        averages = [np.arctanh(value) for value in averages]

        # Calculate the total length (sum of lengths) for correlations at each lag time across all processed files
        # The length of data used for each correlation is important for statistical analysis and confidence interval calculation
        sum_lengths = [sum(sublist[i] for sublist in total_lengths) for i in range(31)]

        # Calculate the standard error for each summed length
        standard_error = [1 / math.sqrt(value) for value in sum_lengths]

        # Set the significance level for statistical analysis
        # Further adjustment by dividing by the number of offsets to correct for the family-wise error rate
        alpha = 0.001 / len(averages)

        # Calculate the Z-value from the normal distribution corresponding to the desired confidence level
        z_value = norm.ppf(1 - alpha / 2)

        # Calculate upper and lower limits of the confidence interval for each averaged correlation value
        # This uses the standard error and Z-value to compute the range around the mean within which the true correlation value is expected to fall
        upper_limits = [avg + z_value * se for avg, se in zip(averages, standard_error)]
        lower_limits = [avg - z_value * se for avg, se in zip(averages, standard_error)]

        # Applying the hyperbolic tangent function to correct the values.
        upper_limits_cor =  [np.tanh(limit) for limit in upper_limits]
        lower_limits_cor =  [np.tanh(limit) for limit in lower_limits]
        averages_cor = [np.tanh(av) for av in averages]

        # Taking the variables to the correlation domain
        upper_limits = upper_limits_cor 
        lower_limits = lower_limits_cor
        averages = averages_cor 

        # Select the color for the current subfolder's plot from the previously generated pastel color palette
        color = colors[color_index]
        plt.figure()

        if args.interpolate:
 
            # If interpolation is enabled, perform cubic spline interpolation for a smooth curve representation
            # Interpolation is done for the averaged correlation values as well as the upper and lower confidence limits
            f_avg = interp1d(shifts, averages, kind='cubic')
            f_lower = interp1d(shifts, lower_limits, kind='cubic')
            f_upper = interp1d(shifts, upper_limits, kind='cubic')
            
            # Generate new, finely spaced shift values for a smoother curve
            shifts_new = np.linspace(min(shifts), max(shifts), 100)

            # Apply the interpolation functions to the new shift values
            averages_smooth = f_avg(shifts_new)
            lower_limits_smooth = f_lower(shifts_new)
            upper_limits_smooth = f_upper(shifts_new)

            # Plot the original and interpolated average correlation values
            # The area between the interpolated upper and lower confidence limits is filled, providing a visual representation of the confidence interval
            plt.plot(shifts, averages, 'o', markersize=3, color = 'gray')
            plt.plot(shifts_new, averages_smooth, '-', color = color)
            plt.fill_between(shifts_new, lower_limits_smooth, upper_limits_smooth, color = color, alpha=0.4)  # Fill area between limits
            # Global (aggregated) plots
            global_ax.plot(shifts, averages, 'o', markersize=3, color = 'gray')
            global_ax.plot(shifts_new, averages_smooth, '-', color = color,label=subfolder)
            global_ax.fill_between(shifts_new, lower_limits_smooth, upper_limits_smooth, color = color, alpha=0.4)

        else:
            # If interpolation is not enabled, plot the average correlation values directly
            # The area between the upper and lower confidence limits is filled similarly
            plt.plot(shifts, averages, 'o-', markersize=3, color='gray')
            plt.plot(shifts, averages, '-', color = color)
            plt.fill_between(shifts, lower_limits, upper_limits, color = color, alpha=0.4)  # Fill area between limits

            # Global (aggregated) plots
            global_ax.plot(shifts, averages, 'o-', markersize=3, color='gray')
            global_ax.plot(shifts, averages, '-', color = color, label=subfolder)
            global_ax.fill_between(shifts, lower_limits, upper_limits, color = color, alpha=0.4)

        # Increment color index for the next subfolder's plots
        color_index += 1
        
        # Set labels and titles for both individual and global (aggregated) plots
        # This includes labeling the axes and providing titles that reflect the content of the plots
        plt.xlabel('Lag [s]')
        plt.ylabel('Pearson Correlation')
        plt.suptitle('Cross-correlation')
        plt.title(subfolder)
        plt.grid(True)

        global_ax.set_xlabel('Lag [s]')
        global_ax.set_ylabel('Pearson Correlation')
        global_ax.set_title('Cross-correlation of all conditions')
        global_ax.grid(True)
        global_ax.legend()
        
        # Save the figures 
        name_fig = os.path.join(folder_path,subfolder + "Mocap.png")  
        plt.savefig(name_fig)

        global_fig_name = os.path.join(folder_path, "All_Conditions_Mocap.png")
        global_fig.savefig(global_fig_name)
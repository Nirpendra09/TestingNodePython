import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math
import numpy as np

def replace_nan_with_average(data):
    """Replaces NaN values in a NumPy array with the average of non-NaN values.

    Args:
        data: A NumPy array.

    Returns:
        A NumPy array with NaN values replaced by the average of non-NaN values.
    """

    # Calculate the average of non-NaN values
    average = np.nanmean(data)

    # Create a mask to identify NaN values
    nan_mask = np.isnan(data)

    # Replace NaN values with the average
    data[nan_mask] = average

    return data

def replace_zeros_with_average(arr):
    """Replaces zeros in a NumPy array with the average of non-zero elements."""
    # Calculate the average of non-zero elements
    avg = np.mean(arr[arr != 0])

    # Create a copy of the array to avoid modifying the original
    new_arr = arr.copy()

    # Replace zeros with the average
    new_arr[new_arr == 0] = avg

    return new_arr

import pandas as pd
import pandas as pd
import numpy as np

import pandas as pd
import numpy as np

def replace_inf_with_neighbor_average_or_global_mean_pd(data, column_name):
    """Replaces inf values in a pandas DataFrame column with the average of neighboring elements or the global mean.

    Args:
        data: A pandas DataFrame.
        column_name: The name of the column to process.

    Returns:
        A pandas DataFrame with inf values replaced in the specified column.
    """

    global_mean = data[column_name].replace([np.inf, -np.inf], np.nan).mean()  # Calculate mean excluding infinities
    inf_indices = data[data[column_name].isin([np.inf, -np.inf])].index
    for i in inf_indices:
        if i == 0:
            data[column_name].loc[i] = data[column_name].loc[i + 1] if not pd.isna(data[column_name].loc[i + 1]) else global_mean
        elif i == len(data) - 1:
            data[column_name].loc[i] = data[column_name].loc[i - 1] if not pd.isna(data[column_name].loc[i - 1]) else global_mean
        else:
            neighbors = [data[column_name].loc[i - 1], data[column_name].loc[i + 1]]
            valid_neighbors = [n for n in neighbors if not pd.isna(n)]
            if len(valid_neighbors) > 0:
                data[column_name].loc[i] = np.mean(valid_neighbors)
            else:
                data[column_name].loc[i] = global_mean

    return data[column_name]

def replace_zero_with_neighbor_average_or_global_mean_pd(data, column_name):
    """Replaces zero values in a pandas DataFrame column with the average of neighboring elements or the global mean.

    Args:
        data: A pandas DataFrame.
        column_name: The name of the column to process.

    Returns:
        A pandas DataFrame with zero values replaced in the specified column.
    """

    global_mean = data[column_name][data[column_name] != 0].mean()  # Calculate mean excluding zeros
    zero_indices = data[data[column_name] == 0].index
    valid_indices = zero_indices[zero_indices < len(data[column_name])]
    zero_indices = valid_indices
    for i in zero_indices:
        if i == 0:
            data[column_name].loc[i] = data[column_name].loc[i + 1] if data[column_name].loc[i + 1] != 0 else global_mean
        elif i == len(data) - 1:
            data[column_name].loc[i] = data[column_name].loc[i - 1] if data[column_name].loc[i - 1] != 0 else global_mean
        else:
            neighbors = [data[column_name].loc[i - 1], data[column_name].loc[i + 1]]
            valid_neighbors = [n for n in neighbors if n != 0]
            if len(valid_neighbors) > 1:
                data[column_name].loc[i] = np.mean(valid_neighbors)
            else:
                data[column_name].loc[i] = global_mean

    return data[column_name]


def replace_nan_with_neighbor_average_or_global_mean_pd(data, column_name):
    """Replaces NaN values in a pandas DataFrame column with the average of neighboring elements or the global mean.

    Args:
        data: A pandas DataFrame.
        column_name: The name of the column to process.

    Returns:
        A pandas DataFrame with NaN values replaced in the specified column.
    """

    global_mean = data[column_name].mean()  # Calculate mean excluding NaNs
    nan_indices = data[data[column_name].isna()].index
    for i in nan_indices:
        if i == 0:
            data[column_name].loc[i] = data[column_name].loc[i + 1] if not pd.isna(data[column_name].loc[i + 1]) else global_mean
        elif i == len(data) - 1:
            data[column_name].loc[i] = data[column_name].loc[i - 1] if not pd.isna(data[column_name].loc[i - 1]) else global_mean
        else:
            neighbors = [data[column_name].loc[i - 1], data[column_name].loc[i + 1]]
            valid_neighbors = [n for n in neighbors if not pd.isna(n)]
            if valid_neighbors.__len__() > 1:
                data[column_name].loc[i] = np.mean(valid_neighbors)
            else:
                data[column_name].loc[i] = global_mean

    return data[column_name]

def replace_zeros_with_neighbor_average_or_global_mean(data):
    """Replaces zero values in a NumPy array with the average of neighboring elements or the global mean.

    Args:
        data: A NumPy array.

    Returns:
        A NumPy array with zero values replaced.
    """
    data = data.to_numpy()
    global_mean = np.mean(data[data != 0])  # Calculate global mean excluding zeros
    zero_indices = np.where(data == 0)[0]
    for i in zero_indices:
        if i == 0:
            data[i] = data[i + 1] if data[i + 1] != 0 else global_mean
        elif i == len(data) - 1:
            data[i] = data[i - 1] if data[i - 1] != 0 else global_mean
        else:
            neighbors = [data[i - 1], data[i + 1]]
            valid_neighbors = [n for n in neighbors if n != 0]
            if valid_neighbors:
                data[i] = np.mean(valid_neighbors)
            else:
                data[i] = global_mean

    return data


def replace_nan_with_neighbor_average_or_global_mean(data):
    """Replaces NaN values in a NumPy array with the average of neighboring elements or the global mean.

    Args:
        data: A NumPy array.

    Returns:
        A NumPy array with NaN values replaced.
    """
    data = data.to_numpy()
    global_mean = np.nanmean(data)  # Calculate global mean excluding NaNs
    nan_indices = np.where(np.isnan(data))[0]
    for i in nan_indices:
        if i == 0:
            data[i] = data[i + 1] if not np.isnan(data[i + 1]) else global_mean
        elif i == len(data) - 1:
            data[i] = data[i - 1] if not np.isnan(data[i - 1]) else global_mean
        else:
            neighbors = [data[i - 1], data[i + 1]]
            valid_neighbors = [n for n in neighbors if not np.isnan(n)]
            if valid_neighbors:
                data[i] = np.mean(valid_neighbors)
            else:
                data[i] = global_mean

    return data


def replace_inf_with_neighbor_average_or_global_mean(data):
    """Replaces inf values in a NumPy array with the average of neighboring elements or the global mean.

    Args:
        data: A NumPy array.

    Returns:
        A NumPy array with inf values replaced.
    """
    data = data.to_numpy()
    global_mean = np.nanmean(data)  # Calculate global mean excluding infinities
    inf_indices = np.where(np.isinf(data))[0]
    for i in inf_indices:
        if i == 0:
            data[i] = data[i + 1] if not np.isinf(data[i + 1]) else global_mean
        elif i == len(data) - 1:
            data[i] = data[i - 1] if not np.isinf(data[i - 1]) else global_mean
        else:
            neighbors = [data[i - 1], data[i + 1]]
            valid_neighbors = [n for n in neighbors if not np.isinf(n)]
            if valid_neighbors:
                data[i] = np.mean(valid_neighbors)
            else:
                data[i] = global_mean

    return data


import numpy as np

def replace_outliers_and_extremes_with_mean(data, method="iqr", threshold=1.5):
    if method == "iqr":
        q25, q75 = np.percentile(data, [10, 90])
        iqr = q75 - q25
        lower_bound = q25 - threshold * iqr
        upper_bound = q75 + threshold * iqr
        outliers = (data < lower_bound) | (data > upper_bound)
    elif method == "zscore":
        z_scores = np.abs((data - np.mean(data)) / np.std(data))
        outliers = z_scores > threshold
    else:
        raise ValueError("Invalid method. Choose 'iqr' or 'zscore'.")
    non_outlier_mean = np.mean(data[~outliers])
    data[outliers] = non_outlier_mean
    return data

def remove_outliers_and_extremes(data, method="iqr", threshold=1.5):
    """Removes outliers and extreme values from a NumPy array.

    Args:
        data: A NumPy array.
        method: The method for outlier detection ("iqr" or "zscore").
        threshold: The threshold for outlier detection (used differently for each method).

    Returns:
        A NumPy array with outliers and extreme values removed.
    """
    data = data.to_numpy()
    if method == "iqr":
        # Calculate interquartile range (IQR)
        q25, q75 = np.percentile(data, [25, 75])
        iqr = q75 - q25

        # Define lower and upper bounds
        lower_bound = q25 - threshold * iqr
        upper_bound = q75 + threshold * iqr

        # Filter data
        data = data[(data >= lower_bound) & (data <= upper_bound)]

    elif method == "zscore":
        # Calculate z-scores
        z_scores = np.abs((data - np.mean(data)) / np.std(data))

        # Filter data
        data = data[z_scores <= threshold]

    else:
        raise ValueError("Invalid method. Choose 'iqr' or 'zscore'.")

    return data
# Load data
# data = pd.read_csv('../Data/2447592223_ACTIVITY.csv')
# data = pd.read_csv('../Data/2508278192_ACTIVITY.csv')
# data = pd.read_csv('../Data/3065620165_ACTIVITY.csv')
data = pd.read_csv('./Data/decoded_fit_data.csv')
# Function to estimate metabolic equivalent (MET)
def met_estimation(speed, grade):
    return 1.0 + 0.03 * speed + 0.00052 * speed**2 + 0.9 * grade


# Convert Time column to datetime format
data['timestamp'] = pd.to_datetime(data['timestamp'])

# Calculate time difference in seconds
data['Time_Diff'] = data['timestamp'].diff().dt.total_seconds()


## Handle missing values _ by Nirpendra
# Assuming data is your DataFrame
if 'speed' not in data.columns:
    data = data.assign(speed=pd.Series([None]*len(data)))
    # Replace None with np.nan
    data['speed'] = data['speed'].apply(lambda x: np.nan if x is None else x)


if 'altitude' not in data.columns:
    data = data.assign(altitude=pd.Series([None]*len(data)))
    # Replace None with np.nan
    data['altitude'] = data['altitude'].apply(lambda x: np.nan if x is None else x)


data['speed'] = replace_nan_with_neighbor_average_or_global_mean(data['speed'])
data['speed'] = replace_inf_with_neighbor_average_or_global_mean(data['speed'])
data['speed'] = replace_zeros_with_neighbor_average_or_global_mean(data['speed'])
# Calculate grade and METs
# Calculate grade, considering time difference

data['altitude_diff'] = data['altitude'].diff()

data['altitude_diff'] = replace_nan_with_neighbor_average_or_global_mean(data['altitude_diff'] )
data['altitude_diff'] = replace_inf_with_neighbor_average_or_global_mean(data['altitude_diff'])

data['Grade'] = data['altitude_diff'] / (data['speed'] * data['Time_Diff'])

# # Count infinities before handling them
num_infinities = np.isinf(data['altitude_diff'].diff()).sum()
print("Number of infinities in altitude:", num_infinities)

num_infinities = np.isnan(data['altitude_diff']).sum()
print("Number of nan in altitude diff:", num_infinities)
num_zeros = np.count_nonzero(data['speed'] == 0)  # 2
print("Number of zeros in speed:", num_zeros)
num_infinities = np.isnan(data['speed']).sum()
print("Number of nan in speed:", num_infinities)
print(np.where(np.isnan(data['speed']))[0])
data['METs'] = data.apply(lambda row: met_estimation(row['speed'], row['Grade']), axis=1)


data['METs'] = replace_nan_with_neighbor_average_or_global_mean(data['METs'] )
data['METs'] = replace_inf_with_neighbor_average_or_global_mean(data['METs'])

# data['METs'] = remove_outliers_and_extremes(data['METs'], method='iqr')

data['heart_rate'] = replace_nan_with_neighbor_average_or_global_mean(data['heart_rate'])
data['heart_rate'] = replace_inf_with_neighbor_average_or_global_mean(data['heart_rate'])
data['heart_rate'] = replace_zeros_with_neighbor_average_or_global_mean(data['heart_rate'])
print("Number of nan in altitude diff:", num_infinities)
num_zeros = np.count_nonzero(data['heart_rate'] == 0)  # 2
print("Number of zeros in heart:", num_zeros)
num_infinities = np.isnan(data['heart_rate']).sum()
print("Number of nan in heart:", num_infinities)
print(np.where(np.isnan(data['heart_rate']))[0])

# Approximate RMSSD
data['HR_diff'] = data['heart_rate'].diff()  # Calculate successive heart rate differences
data['HR_diff_sq'] = data['HR_diff']**2      # Square the differences
rmssd_window_size = 30  # Adjust window size as needed
data['RMSSD_approx'] = data['HR_diff_sq'].rolling(window=rmssd_window_size).mean()**0.5


# Incorporate heart rate variability (HRV)
# Example using rolling window standard deviation of heart rate
window_size = 10
data['HRV_Approx'] = data['heart_rate'].rolling(window=window_size).std()
data['HRV_Factor'] = 1 / (data['HRV_Approx'] + 1)

# Calculate effort level
data['Effort_Level'] = data['METs'] * data['heart_rate'] * data['HRV_Factor']

# data['Effort_Level'] = data['METs'] * data['heart_rate']


data['Effort_Level'] = replace_nan_with_neighbor_average_or_global_mean(data['Effort_Level'] )
data['Effort_Level'] = replace_inf_with_neighbor_average_or_global_mean(data['Effort_Level'])

data['Effort_Level'] = replace_outliers_and_extremes_with_mean(data['Effort_Level'].to_numpy(), method="iqr")


# # Count infinities before handling them
# num_infinities = np.isinf(data['Effort_Level']).sum()
# print("Number of infinities in Effort_Level:", num_infinities)
#
# Handle infinities (as shown in the previous example)
# data['Effort_Level'] = pd.to_numeric(data['Effort_Level'], errors='coerce')



# data['Effort_Level'] = np.where(np.isnan(data['Effort_Level']), 10, data['Effort_Level'])


# Count infinities before handling them
num_infinities = np.isinf(data['Effort_Level']).sum()
print("Number of infinities in Effort_Level:", num_infinities)

num_infinities = np.isnan(data['Effort_Level']).sum()
print("Number of nans in Effort_Level:", num_infinities)
# Analyze effort distribution
effort_variation = data['Effort_Level'].max() - data['Effort_Level'].min()
if effort_variation < 500:
    print("Effort distribution: Steady")
else:
    print("Effort distribution: Variable with bursts and recoveries")

# Visualization
plt.scatter(data['timestamp'], data['Effort_Level'])
plt.xlabel('Time')
plt.ylabel('Effort Level (METs * HR)')
plt.title('Effort Distribution Throughout the Run')
plt.show()



# Load data

# ... (MET estimation and effort level calculation as before) ...

# Training Load Estimation

# 1. Duration-Based Approach:
duration_minutes = len(data) / 60
average_effort = data['Effort_Level'].mean()
training_load_duration_based = duration_minutes * average_effort

# 2. Impulse-Based Approach (Simplified TRIMP):
trimp_factor = 0.64 * math.exp(1.92 * data['Effort_Level'].mean() / data['heart_rate'].max())
training_load_trimp = duration_minutes * trimp_factor


sample_points = data.reset_index()['timestamp'].diff().dt.total_seconds().fillna(0).cumsum().values

# 3. Area Under the Curve (AUC):
effort_auc = np.trapz(y=data['Effort_Level'], x=sample_points)

# Optional: Normalize AUC by duration
duration_seconds = data['timestamp'].max() - data['timestamp'].min()
effort_auc_normalized_time = effort_auc / duration_seconds.seconds


max_heart_rate = data['heart_rate'].max()
mean_heart_rate = data['heart_rate'].mean()
# 3. Estimate theoretical maximum effort level (assuming max HR throughout)
max_effort_level = data['METs'] * max_heart_rate  # Assumes METs remain the same

mean_effort_level = data['METs'] * mean_heart_rate
mean_effort_auc = np.trapz(y = mean_effort_level, x=sample_points)
# 4. Calculate theoretical maximum AUC
max_effort_auc = np.trapz(y=max_effort_level, x=sample_points)

# 5. Normalize actual AUC by theoretical maximum AUC
normalized_auc_with_max = effort_auc / max_effort_auc
normalized_auc_with_mean = effort_auc / mean_effort_auc
# Print or use the training load estimates as needed
print("Duration-Based Training Load:", training_load_duration_based)
print("TRIMP-Based Training Load:", training_load_trimp)
print("AUC-Based Training Load:", effort_auc)
print("Normalized AUC-Based Training Load:", effort_auc_normalized_time)
print("Normalized Load with Max: ", normalized_auc_with_max)
print("Normalized Load with Mean: ", normalized_auc_with_mean)
print("total duration in minutes: ", duration_seconds.seconds//60)
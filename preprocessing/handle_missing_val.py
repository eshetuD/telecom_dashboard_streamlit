import pandas as pd

#fill category data
def fill_missing_with_mode(data, columns):
    for column in columns:
        data[column] = data[column].fillna(data[column].mode()[0])
    return data

#fill numeric data
def fill_missing_with_mean(data):
    # Convert columns to numeric, coercing errors to NaN
    numeric_df = data.apply(pd.to_numeric, errors='coerce')

    # Calculate the mean of each column
    mean_values = numeric_df.mean()

    # Fill missing values with the mean of each column
    filled_data_frame = numeric_df.fillna(mean_values)

    return filled_data_frame
def aggregate_user(df):
    user_aggregated = df.groupby('MSISDN/Number').agg({
    'MSISDN/Number': 'count',                    # Number of xDR sessions
    'Dur. (ms).1': 'sum',                    # Session duration
    'Total DL (Bytes)': 'sum',               # Total download data
    'Total UL (Bytes)': 'sum',               # Total upload data
    'Social Media DL (Bytes)': 'sum',        # Total social media download data
    'Social Media UL (Bytes)': 'sum',        # Total social media upload data
    'Google DL (Bytes)': 'sum',              # Total Google download data
    'Google UL (Bytes)': 'sum',              # Total Google upload data
    'Email DL (Bytes)': 'sum',               # Total email download data
    'Email UL (Bytes)': 'sum',               # Total email upload data
    'Youtube DL (Bytes)': 'sum',             # Total YouTube download data
    'Youtube UL (Bytes)': 'sum',             # Total YouTube upload data
    'Netflix DL (Bytes)': 'sum',             # Total Netflix download data
    'Netflix UL (Bytes)': 'sum',             # Total Netflix upload data
    'Gaming DL (Bytes)': 'sum',              # Total gaming download data
    'Gaming UL (Bytes)': 'sum',              # Total gaming upload data
    'Other DL (Bytes)': 'sum',               # Total other download data
    'Other UL (Bytes)': 'sum'                # Total other upload data
    })

    # Rename columns for clarity
    user_aggregated = user_aggregated.rename(columns={
        'MSISDN/Number': 'Number_of_xDR_sessions',
        'Dur. (ms).1': 'Session_duration',
        'Total DL (Bytes)': 'Total_DL_data',
        'Total UL (Bytes)': 'Total_UL_data',
        'Social Media DL (Bytes)': 'Social_Media_DL_data',
        'Social Media UL (Bytes)': 'Social_Media_UL_data',
        'Google DL (Bytes)': 'Google_DL_data',
        'Google UL (Bytes)': 'Google_UL_data',
        'Email DL (Bytes)': 'Email_DL_data',
        'Email UL (Bytes)': 'Email_UL_data',
        'Youtube DL (Bytes)': 'Youtube_DL_data',
        'Youtube UL (Bytes)': 'Youtube_UL_data',
        'Netflix DL (Bytes)': 'Netflix_DL_data',
        'Netflix UL (Bytes)': 'Netflix_UL_data',
        'Gaming DL (Bytes)': 'Gaming_DL_data',
        'Gaming UL (Bytes)': 'Gaming_UL_data',
        'Other DL (Bytes)': 'Other_DL_data',
        'Other UL (Bytes)': 'Other_UL_data'
    })

    # Display the aggregated information per user
    user_aggregated.sort_values(by="Number_of_xDR_sessions", ascending=False)
    return user_aggregated
import plotly.express as px
import pandas as pd
import streamlit as st

def plot_top_handsets(df, top_n=10):
    """
    Plot the top N handsets used by customers.

    Parameters:
    - df: DataFrame containing the relevant data
    - top_n: Number of top handsets to display (default is 10)
    """
    handset_counts = df['Handset Type'].value_counts().head(top_n).sort_values(ascending=True)

    fig = px.bar(handset_counts, orientation='h', title=f'Top {top_n} Handsets Used by Customers')
    return fig

def plot_top_manufacturers(df, top_n=3):
    """
    Plot the top N handset manufacturers.

    Parameters:
    - df: DataFrame containing the relevant data
    - top_n: Number of top manufacturers to display (default is 3)
    """
    top_manufacturers = df['Handset Manufacturer'].value_counts().head(top_n)

    fig_manufacturers = px.bar(
        top_manufacturers,
        x=top_manufacturers.index,
        y=top_manufacturers.values,
        title=f'Top {top_n} Handset Manufacturers'
    )

    return fig_manufacturers

def plot_top_handsets_per_manufacturer(df, top_n_manufacturers=3, top_n_handsets=5):
    """
    Plot the top N handsets per top M manufacturers using Plotly Express.

    Parameters:
    - df: DataFrame containing the relevant data
    - top_n_manufacturers: Number of top manufacturers to consider (default is 3)
    - top_n_handsets: Number of top handsets per manufacturer to display (default is 5)
    """
    top_manufacturers = df['Handset Manufacturer'].value_counts().head(top_n_manufacturers).index.tolist()

    top_handsets_df = df.groupby(['Handset Manufacturer', 'Handset Type']).size().reset_index(name='count')
    top_handsets_df = top_handsets_df[top_handsets_df['Handset Manufacturer'].isin(top_manufacturers)]
    top_handsets_df = top_handsets_df.sort_values(by='count', ascending=False).groupby('Handset Manufacturer').head(top_n_handsets)

    fig_top_handsets = px.bar(
        top_handsets_df,
        x='Handset Type',
        y='count',
        color='Handset Manufacturer',
        title=f'Top {top_n_handsets} Handsets per Top {top_n_manufacturers} Manufacturers'
    )
    fig_top_handsets.update_layout(xaxis_title='Handset Type', yaxis_title='Count', barmode='group')
    return fig_top_handsets

def plot_top_app_usage(df):
    applications_columns = ['Social Media DL (Bytes)', 'Google DL (Bytes)', 'Email DL (Bytes)',
                         'Youtube DL (Bytes)', 'Netflix DL (Bytes)', 'Gaming DL (Bytes)', 'Other DL (Bytes)',
                         'Social Media UL (Bytes)', 'Google UL (Bytes)', 'Email UL (Bytes)',
                         'Youtube UL (Bytes)', 'Netflix UL (Bytes)', 'Gaming UL (Bytes)', 'Other UL (Bytes)']

    # Assuming df_clean is your DataFrame
    # Load your DataFrame or replace df_clean with your actual DataFrame

    # Create a new DataFrame with the total data volume for each application
    total_data_per_application = pd.DataFrame({
        'Application': applications_columns,
        'Total_Data_Volume': df[applications_columns].sum()
    })

    # Sort the DataFrame by total data volume in descending order
    total_data_per_application = total_data_per_application.sort_values(by='Total_Data_Volume', ascending=True)

    # Create a Plotly bar chart
    fig = px.bar(total_data_per_application, x='Total_Data_Volume', y='Application', orientation='h',
                title='Total Data Volume for Each Application',
                labels={'Total_Data_Volume': 'Total Data Volume (Bytes)', 'Application': 'Application'},
                color='Total_Data_Volume',
                color_continuous_scale='viridis')

    # Display the chart in the Streamlit app
    return fig
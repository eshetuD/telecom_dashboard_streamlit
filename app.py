import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from streamlit_card import card
import psycopg2

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import euclidean_distances

from src.dbconn import connect_db
from preprocessing.handle_missing_val import fill_missing_with_mode, fill_missing_with_mean
from preprocessing.user_aggregation import aggregate_user
from preprocessing.user_analysis import plot_top_handsets, plot_top_manufacturers ,plot_top_handsets_per_manufacturer, plot_top_app_usage

st.set_page_config(
    page_title="Telecom Dashboard",
    page_icon=":bar_chart:",
    layout="wide"
)

df= connect_db()


# fill category column with mode
category_data = ['Start','End','Last Location Name', 'Handset Manufacturer','Handset Type']
category_data1 = df[['Start','End','Last Location Name', 'Handset Manufacturer','Handset Type']]
fill_missing_with_mode(category_data1, category_data)
#df['End'] = fill_missing_with_mode(df, "End")
#df['Last Location Name'] = fill_missing_with_mode(df, "Last Location Name")
#df['Handset Manufacturer'] = fill_missing_with_mode(df, "Handset Manufacturer")
#df['Handset Type'] = fill_missing_with_mode(df, "Handset Type")




#drop category data from df to fill numeric data
df_fill = df.drop(['Start','End','Last Location Name', 'Handset Manufacturer','Handset Type'], axis=1)

#Drop a feature with more than 70000 missing Values
df_fill1 = df_fill.drop(['Nb of sec with 125000B < Vol DL', 'Nb of sec with 1250B < Vol UL < 6250B','Nb of sec with 31250B < Vol DL < 125000B','Nb of sec with 37500B < Vol UL','Nb of sec with 6250B < Vol DL < 31250B','Nb of sec with 6250B < Vol UL < 37500B','HTTP DL (Bytes)','HTTP UL (Bytes)'], axis=1) 

#Fill Numeric value
df_fill_numeric = fill_missing_with_mean(df_fill1)

#concatenet two dataframe 
df = pd.concat([df_fill_numeric, category_data1], axis=1, join="inner")



# Assuming 'MSISDN/Number' is the column containing user identifiers
# and columns like 'Social Media DL (Bytes)', 'Google DL (Bytes)', etc. represent data volume for each application
applications_columns = ['Social Media DL (Bytes)', 'Google DL (Bytes)', 'Email DL (Bytes)',
                         'Youtube DL (Bytes)', 'Netflix DL (Bytes)', 'Gaming DL (Bytes)', 'Other DL (Bytes)',
                         'Social Media UL (Bytes)', 'Google UL (Bytes)', 'Email UL (Bytes)',
                         'Youtube UL (Bytes)', 'Netflix UL (Bytes)', 'Gaming UL (Bytes)', 'Other UL (Bytes)']

# Create a new DataFrame with the total data volume for each application
total_data_per_application = pd.DataFrame({
    'Application': applications_columns,
    'Total_Data_Volume': df[applications_columns].sum()
})

# Find the application with the highest total data volume
most_used_application = total_data_per_application.loc[total_data_per_application['Total_Data_Volume'].idxmax()]

# Display the result
app = most_used_application['Application']
gb = (((most_used_application['Total_Data_Volume'])/1024)/1024)/1024


st.title(":bar_chart: Telecom Dashboard")
st.markdown("##")

total_DL = int(df["Total DL (Bytes)"].sum())
total_UL = int(df["Total UL (Bytes)"].sum())
#st.metric(label="Total Bytes of Download", value=(f"GB {round(((total_DL/1024)/1024)/1024,2)}"))
left_column, middle_column, right_column = st.columns(3)
with left_column:
    st.subheader("Total Download")
    st.subheader(f"{round(((total_DL/1024)/1024)/1024,2)} GB")
with middle_column:
    st.subheader("Total Upload")
    st.subheader(f"GB {round(((total_UL/1024)/1024)/1024,2)}")
with right_column:
    st.subheader(f"Application more used: {app}")
    st.subheader(f"GB {round(gb,2)}")

st.markdown("---")

# Function to remove grid lines and set background color to transparent
def style_axes(ax):
    ax.set_facecolor('none')  # Set background color to transparent
    ax.grid(False)  # Remove grid lines
    return ax


# Plot the top 10 handsets countplot using seaborn
#fig_handsets = px.bar(df['Handset Type'].value_counts().head(10).sort_values(ascending=True), orientation='h', title='Top 10 Handsets Used by Customers')


# Plot the top 3 handset manufacturers
#top_manufacturers = df['Handset Manufacturer'].value_counts().head(3)
#fig_manufacturers = px.bar(top_manufacturers, x=top_manufacturers.index, y=top_manufacturers.values, title='Top 3 Handset Manufacturers')


#df_combined = df[df['Handset Manufacturer'].isin(top_manufacturers.index)].sort_values(by='Handset Manufacturer', ascending=True).head(5)
#fig_combined = px.bar(df_combined, x='Handset Type', color='Handset Manufacturer', title='Top 5 Handsets per Top 3 Manufacturers')

# Group by 'Handset Manufacturer' and 'Handset Type', then find the top 5 handsets for each manufacturer
top_manufacturer_handsets = df.groupby('Handset Manufacturer')['Handset Type'].value_counts().groupby(level=0, group_keys=False).nlargest(5)

# Convert the result to a DataFrame
top_manufacturer_handsets_df = top_manufacturer_handsets.reset_index(name='Count')

# Plot the interactive bar chart using Plotly Express

fig = px.bar(
        top_manufacturer_handsets_df,
        x='Handset Type',
        y='Count',
        color='Handset Manufacturer',
        title='Top 5 Handsets per Top 3 Manufacturers',
        labels={'Handset Type': 'Handset Type', 'Count': 'Count'},
        category_orders={'Handset Type': top_manufacturer_handsets_df['Handset Type'].value_counts().head(5).index},
    )

fig.update_layout(
        barmode='group',
        xaxis=dict(title='Handset Type'),
        yaxis=dict(title='Count'),
        legend=dict(title='Handset Manufacturer', orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
    )
#####

# Plot the top 5 handsets per top 3 manufacturers using Plotly Express
top_manufacturers = df['Handset Manufacturer'].value_counts().head(3).index.tolist()
top_handsets = df.groupby(['Handset Manufacturer', 'Handset Type']).size().reset_index(name='count')
top_handsets = top_handsets[top_handsets['Handset Manufacturer'].isin(top_manufacturers)]
top_handsets = top_handsets.sort_values(by='count', ascending=False).groupby('Handset Manufacturer').head(5)

fig_top_handsets = px.bar(top_handsets, x='Handset Type', y='count', color='Handset Manufacturer',
                          title='Top 5 Handsets per Top 3 Manufacturers')
fig_top_handsets.update_layout(xaxis_title='Handset Type', yaxis_title='Count', barmode='group')

######



left_column, middle_column, right_column = st.columns(3)
with left_column:
    #st.subheader('Top 10 Handsets Used by Customers')
    # Display the plot in Streamlit
    fig_handsets = plot_top_handsets(df)
    st.plotly_chart(fig_handsets)
with middle_column:
    #st.subheader('Top 3 Handset Manufacturers')
    fig_manufacturers= plot_top_manufacturers(df)
    st.plotly_chart(fig_manufacturers)
with right_column:
    #st.subheader('Top 5 Handsets per Top 3 Manufacturers')
    #st.plotly_chart(fig_combined)
    figmh = plot_top_handsets_per_manufacturer(df)
    st.plotly_chart(figmh)
    
st.markdown("---")


left_column, middle_column, right_column = st.columns(3)
with left_column:
    #st.subheader('Top 10 Handsets Used by Customers')
    # Display the plot in Streamlit
    st.subheader("Top xDR Sessions & Duration Per User")
    # Assuming 'MSISDN/Number' is the column containing user identifiers and 'Dur. (ms)' is used as the session duration
    user_session_duration = df.groupby('MSISDN/Number')['Dur. (ms)'].sum().reset_index()

    # Rename the columns for clarity
    user_session_duration.columns = ['MSISDN/Number', 'Total_Session_Duration']

    # Display the result
    st.table(user_session_duration.head(10))
with middle_column:
    st.subheader("Top DL and UL Per User")
    user_total_data = df.groupby('MSISDN/Number')[['Total DL (Bytes)', 'Total UL (Bytes)']].sum().reset_index()

    # Rename the columns for clarity
    user_total_data.columns = ['MSISDN/Number', 'Total_DL_Data', 'Total_UL_Data']

    # Display the result
    st.table(user_total_data.head(10))
with right_column:
    #st.subheader('Top 5 Handsets per Top 3 Manufacturers')
    #st.plotly_chart(fig_combined)
    plot_top_app_usage = plot_top_app_usage(df)
    st.plotly_chart(plot_top_app_usage)
    
st.markdown("---")

left_column, middle_column, right_column = st.columns(3)
with left_column:
    st.subheader("Session Frequency")
    sessions_df = df[['MSISDN/Number', 'Start', 'End']]

    # Convert 'Start' and 'End' columns to datetime format
    sessions_df['Start'] = pd.to_datetime(sessions_df['Start'])
    sessions_df['End'] = pd.to_datetime(sessions_df['End'])

    # Calculate session duration in minutes
    sessions_df['Session_Duration_Minutes'] = (sessions_df['End'] - sessions_df['Start']).dt.total_seconds() / 60

    # Calculate the number of sessions for each user
    session_frequency = sessions_df.groupby('MSISDN/Number')['Start'].count().reset_index()
    session_frequency.columns = ['MSISDN/Number', 'Session_Frequency']

    # Display the result
    

    # Display the result
    st.table(session_frequency.sort_values(by="Session_Frequency", ascending=False).head(10))
with middle_column:
    sessions_df = df[['MSISDN/Number', 'Start', 'End']]

    # Convert 'Start' and 'End' columns to datetime format
    sessions_df['Start'] = pd.to_datetime(sessions_df['Start'])
    sessions_df['End'] = pd.to_datetime(sessions_df['End'])

    # Calculate session duration in minutes
    sessions_df['Session_Duration_Minutes'] = (sessions_df['End'] - sessions_df['Start']).dt.total_seconds() / 60

    # Create a Plotly histogram
    fig = px.histogram(sessions_df, x='Session_Duration_Minutes', nbins=30,
                    title='Session Duration Distribution',
                    labels={'Session_Duration_Minutes': 'Session Duration (Minutes)', 'count': 'Number of Sessions'},
                    color_discrete_sequence=['skyblue'])

    # Display the chart in the Streamlit app
    st.plotly_chart(fig)
with right_column:
    sessions_df = df[['MSISDN/Number', 'Total DL (Bytes)', 'Total UL (Bytes)']]

    # Calculate the total traffic (download + upload) for each session
    sessions_df['Total_Traffic_Bytes'] = sessions_df['Total DL (Bytes)'] + sessions_df['Total UL (Bytes)']

    # Create a Plotly histogram
    fig = px.histogram(sessions_df, x='Total_Traffic_Bytes', nbins=30,
                    title='Total Traffic Distribution',
                    labels={'Total_Traffic_Bytes': 'Total Traffic (Bytes)', 'count': 'Number of Sessions'},
                    color_discrete_sequence=['skyblue'])

    # Display the chart in the Streamlit app
    st.plotly_chart(fig)
    
st.markdown("---")

st.subheader("Aggregate network parameters per customer")
@st.cache_data
def aggre_user():
    network_data = df[['MSISDN/Number', 'TCP DL Retrans. Vol (Bytes)', 'TCP UL Retrans. Vol (Bytes)',
                    'Avg RTT DL (ms)', 'Avg RTT UL (ms)', 'Handset Type', 'Avg Bearer TP DL (kbps)',
                    'Avg Bearer TP UL (kbps)']]

        # Aggregate network parameters per customer
    user_aggregated_network = network_data.groupby('MSISDN/Number').agg({
            'TCP DL Retrans. Vol (Bytes)': lambda x: x.mean(),
            'TCP UL Retrans. Vol (Bytes)': lambda x: x.mean(),
            'Avg RTT DL (ms)': lambda x: x.mean(),
            'Avg RTT UL (ms)': lambda x: x.mean(),
            'Handset Type': lambda x: x.mode().iloc[0] if not x.mode().empty else None,
            'Avg Bearer TP DL (kbps)': lambda x: x.mean(),
            'Avg Bearer TP UL (kbps)': lambda x: x.mean()
        })

        # Rename columns for clarity
    user_aggregated_network = user_aggregated_network.rename(columns={
            'TCP DL Retrans. Vol (Bytes)': 'Avg_TCP_Retrans_DL',
            'TCP UL Retrans. Vol (Bytes)': 'Avg_TCP_Retrans_UL',
            'Avg RTT DL (ms)': 'Avg_RTT_DL',
            'Avg RTT UL (ms)': 'Avg_RTT_UL',
            'Avg Bearer TP DL (kbps)': 'Avg_Throughput_DL',
            'Avg Bearer TP UL (kbps)': 'Avg_Throughput_UL'
        })
    return user_aggregated_network
user_aggregated_network = aggre_user()
st.table(user_aggregated_network.head())


st.markdown("---")



left_column, middle_column, right_column = st.columns(3)
with left_column:
    # Select the relevant columns for clustering
    experience_metrics = df[['Avg Bearer TP DL (kbps)', 'Avg Bearer TP UL (kbps)',
                            'TCP DL Retrans. Vol (Bytes)', 'TCP UL Retrans. Vol (Bytes)']]

    # Fill any missing values with the mean
    experience_metrics = experience_metrics.fillna(experience_metrics.mean())

    # Standardize the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(experience_metrics)

    # Perform k-means clustering with k=3
    kmeans = KMeans(n_clusters=3, random_state=42)
    df['Cluster'] = kmeans.fit_predict(scaled_data)

    # Analyze the cluster centers
    cluster_centers = pd.DataFrame(scaler.inverse_transform(kmeans.cluster_centers_), columns=experience_metrics.columns)

    # Display the original data used for clustering
    st.subheader('Original Data for Clustering')
    st.table(experience_metrics.head())
with middle_column:
    # Display the cluster assignments
    st.subheader('Cluster Assignments')
    st.table(df[['Avg Bearer TP DL (kbps)', 'Avg Bearer TP UL (kbps)',
                                'TCP DL Retrans. Vol (Bytes)', 'TCP UL Retrans. Vol (Bytes)', 'Cluster']].head())
with right_column:
    # Display the cluster centers
    st.subheader('Cluster Centers')
    st.table(cluster_centers.head())

####### TASK 5 ####################################

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import pairwise_distances_argmin_min

# Select relevant columns for engagement and experience analysis
engagement_experience_metrics = df[['Avg Bearer TP DL (kbps)', 'Avg Bearer TP UL (kbps)',
                                    'TCP DL Retrans. Vol (Bytes)', 'TCP UL Retrans. Vol (Bytes)']]

# Fill any missing values with the mean
engagement_experience_metrics = engagement_experience_metrics.fillna(engagement_experience_metrics.mean())

# Standardize the data
scaler = StandardScaler()
scaled_metrics = scaler.fit_transform(engagement_experience_metrics)

# Assign engagement and experience scores
engagement_scores = pairwise_distances_argmin_min(scaled_metrics, kmeans.cluster_centers_[0].reshape(1, -1))[1]
experience_scores = pairwise_distances_argmin_min(scaled_metrics, cluster_centers.iloc[cluster_centers['Avg Bearer TP DL (kbps)'].idxmin()].values.reshape(1, -1))[1]

# Add scores to the DataFrame
df['Engagement Score'] = engagement_scores
df['Experience Score'] = experience_scores

#Task 5.2
# Calculate the average satisfaction score
df['Satisfaction Score'] = df[['Engagement Score', 'Experience Score']].mean(axis=1)

#Task 5.3

# Select features for the regression model
features = ['Avg Bearer TP DL (kbps)', 'Avg Bearer TP UL (kbps)',
            'TCP DL Retrans. Vol (Bytes)', 'TCP UL Retrans. Vol (Bytes)']

# Fill any missing values with the mean
df[features] = df[features].fillna(df[features].mean())

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df[features], df['Satisfaction Score'], test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize the linear regression model
model = LinearRegression()

# Train the model
model.fit(X_train_scaled, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test_scaled)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
###print score result#####


#Task 5.4
# Select relevant columns for clustering
engagement_experience_scores = df[['Engagement Score', 'Experience Score']]

# Fill any missing values with the mean
engagement_experience_scores = engagement_experience_scores.fillna(engagement_experience_scores.mean())

# Standardize the data
scaler = StandardScaler()
scaled_scores = scaler.fit_transform(engagement_experience_scores)


# Perform k-means clustering with k=2
kmeans_2 = KMeans(n_clusters=2, random_state=42)
df['Cluster_2'] = kmeans_2.fit_predict(scaled_scores)


##Task 5.5
cluster_aggregated_scores = df.groupby('Cluster_2').agg({
    'Satisfaction Score': 'mean',
    'Experience Score': 'mean'
}).reset_index()

#Task 5.5
st.subheader("Aggregate of average Engagement, Experience and Satisfaction score per user")
st.table(df[['MSISDN/Number', 'Engagement Score', 'Experience Score', 'Satisfaction Score']].sort_values(by='Satisfaction Score',ascending=False).head())


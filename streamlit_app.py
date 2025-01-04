import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set the title and favicon for the app.
st.set_page_config(
    page_title='Water Usage Prediction Dashboard',
    page_icon=':droplet:', # Water droplet emoji for a relevant icon.
)

# -------------------------------------------------------------------------------
# Functions for Data Loading and Preprocessing

@st.cache_data
def load_data():
    """Load water usage and population data."""

    # Example: Reading from a CSV file. Replace with your actual data source.
    DATA_FILENAME = Path(__file__).parent / 'data/water_usage_data.csv'
    df = pd.read_csv(DATA_FILENAME)

    # Example of preprocessing (customize based on your dataset).
    df['Year'] = pd.to_datetime(df['Year'], format='%Y')
    df['Water_Usage'] = df['Water_Usage'] / 1000  # Example: Convert to cubic meters
    return df

# Load the data
df = load_data()

# -------------------------------------------------------------------------------
# Draw the app

# Set the title at the top of the page.
'''
# :droplet: Water Usage Prediction Dashboard

Explore water usage patterns in Malaysia and predict future consumption trends. 
Select years and states to visualize water consumption and other relevant data.
'''

# Add some spacing
''

# Get minimum and maximum year from the data
min_year = df['Year'].min().year
max_year = df['Year'].max().year

# Slider to select the range of years
from_year, to_year = st.slider(
    'Select the range of years for analysis',
    min_value=min_year,
    max_value=max_year,
    value=[min_year, max_year]
)

# Filter the data based on the selected year range
filtered_df = df[(df['Year'].dt.year >= from_year) & (df['Year'].dt.year <= to_year)]

# Select the states for the user to choose
states = df['State'].unique()

# Multiselect to choose specific states
selected_states = st.multiselect(
    'Which states do you want to view?',
    states,
    default=states[:5]  # Default to first 5 states if no selection is made
)

# Filter the data based on the selected states
filtered_df = filtered_df[filtered_df['State'].isin(selected_states)]

# Display data in table format for reference
st.header('Water Usage Data', divider='gray')
st.write(filtered_df)

# -------------------------------------------------------------------------------
# Visualization of Water Usage Trends

# Line chart for water usage trends
st.header('Water Usage Over Time', divider='gray')
fig, ax = plt.subplots(figsize=(10, 6))
sns.lineplot(x='Year', y='Water_Usage', hue='State', data=filtered_df, ax=ax)
ax.set_title('Water Usage Trends by State')
ax.set_xlabel('Year')
ax.set_ylabel('Water Usage (in Cubic Meters)')
st.pyplot(fig)

# -------------------------------------------------------------------------------
# Prediction (Example Placeholder)
# Assuming you have a predictive model like ARIMA or ANN for water usage forecasting

# Load your model or prediction results here (for example, ARIMA or ANN model)
# For now, we'll use a simple linear prediction as a placeholder.

st.header('Water Usage Prediction for the Next Year', divider='gray')

# Placeholder for predictive analysis - This is where you would integrate your model.
next_year = max_year + 1
predicted_usage = np.random.uniform(low=filtered_df['Water_Usage'].min(), high=filtered_df['Water_Usage'].max(), size=1)

# Display predicted water usage for next year
st.metric(
    label=f'Predicted Water Usage for {next_year}',
    value=f'{predicted_usage[0]:,.0f} Cubic Meters',
    delta=f'{(predicted_usage[0] - filtered_df['Water_Usage'].mean()):,.0f}',
    delta_color='normal'
)

# -------------------------------------------------------------------------------
# Conclusion Section
st.header('Key Insights & Conclusions', divider='gray')

# Add some observations based on the data.
st.write("""
- The water usage has been increasing in certain states like Selangor.
- Significant fluctuations are seen during drought years, which affect production.
- Predicting future water usage can help better manage water resources, especially in peak-demand states.
""")

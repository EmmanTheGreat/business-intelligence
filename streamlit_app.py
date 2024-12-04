import streamlit as st
import pandas as pd
import plost as pl
import altair as alt
import pydeck as pdk
import plotly.express as px
import numpy as np
import time
import folium
import plotly.graph_objects as go
from folium import Map
from folium.plugins import HeatMap
from geopy.geocoders import Nominatim, GoogleV3
from geopy.exc import GeocoderTimedOut, GeocoderInsufficientPrivileges
from streamlit_folium import st_folium

# Configure the page
st.set_page_config(layout='wide', initial_sidebar_state='expanded')

# Apply custom styles
try:
    with open('style.css') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
except FileNotFoundError:
    st.warning("Custom CSS file not found. Default styling applied.")

# Load the data
@st.cache_data
def load_data():
    data = pd.read_csv("clean_data.csv", parse_dates=["Order Date"])
    return data

sales_data = load_data()

#update


# Sidebar configuration
st.sidebar.markdown('<h2 class="sales-dashboard-header">Sales Dashboard</h2>', unsafe_allow_html=True)


st.sidebar.subheader('Filter Options')

# Filter by date range
date_range = st.sidebar.date_input(
    "Select Date Range",
    [sales_data['Order Date'].min(), sales_data['Order Date'].max()]
)
filtered_data = sales_data[
    (sales_data['Order Date'] >= pd.to_datetime(date_range[0])) &
    (sales_data['Order Date'] <= pd.to_datetime(date_range[1]))
]

# Filter by product
st.sidebar.markdown('<div class="filter-container"><div class="filter-header">📦 Select Product(s)</div>', unsafe_allow_html=True)
product_filter = st.sidebar.multiselect('', sales_data['Product Name'].unique())
if product_filter:
    filtered_data = filtered_data[filtered_data['Product Name'].isin(product_filter)]
else:
    st.sidebar.warning("No products selected. Showing all data.")
st.sidebar.markdown('</div>', unsafe_allow_html=True)

# Filter by state
st.sidebar.markdown('<div class="filter-container"><div class="filter-header">🌍 Select State(s)</div>', unsafe_allow_html=True)
state_filter = st.sidebar.multiselect('', sales_data['State'].unique())
if state_filter:
    filtered_data = filtered_data[filtered_data['State'].isin(state_filter)]
else:
    st.sidebar.warning("No states selected. Showing all data.")
st.sidebar.markdown('</div>', unsafe_allow_html=True)

# Filter by ship mode
st.sidebar.markdown('<div class="filter-container"><div class="filter-header">🚛 Select Ship Mode(s)</div>', unsafe_allow_html=True)
ship_mode_filter = st.sidebar.multiselect('', sales_data['Ship Mode'].unique())
if ship_mode_filter:
    filtered_data = filtered_data[filtered_data['Ship Mode'].isin(ship_mode_filter)]
else:
    st.sidebar.warning("No ship modes selected. Showing all data.")
st.sidebar.markdown('</div>', unsafe_allow_html=True)

# Filter by category
st.sidebar.markdown('<div class="filter-container"><div class="filter-header">🛍️ Select Category(ies)</div>', unsafe_allow_html=True)
category_filter = st.sidebar.multiselect('', sales_data['Category'].unique())
if category_filter:
    filtered_data = filtered_data[filtered_data['Category'].isin(category_filter)]
else:
    st.sidebar.warning("No categories selected. Showing all data.")
st.sidebar.markdown('</div>', unsafe_allow_html=True)

# Heatmap and donut chart parameters
st.sidebar.subheader('Heatmap Parameter')
time_hist_color = st.sidebar.selectbox('Color by', ('Sales',))

st.sidebar.subheader('Donut Chart Parameter')
donut_theta = st.sidebar.selectbox('Select Data', ('Sales', 'Product ID'))

st.sidebar.markdown('''
---
Credits to Pixegami and Data Professor for the Concept.
''')

# Main dashboard
st.markdown('<h1 class="key-metrics-header">Key Metrics</h1>', unsafe_allow_html=True)

col1, col2, col3, col4, col5, col6 = st.columns(6)

# Metrics Calculation
if not filtered_data.empty:
    # Total Sales
    total_sales = filtered_data['Sales'].sum()
    prev_total_sales = filtered_data[filtered_data['Order Date'] < (filtered_data['Order Date'].max() - pd.DateOffset(years=1))]['Sales'].sum()
    total_sales_rate = (total_sales - prev_total_sales) / prev_total_sales * 100 if prev_total_sales else 0
    
    # Top Product
    top_product = filtered_data.groupby('Product Name')['Sales'].sum().idxmax()
    prev_top_product_sales = filtered_data[filtered_data['Order Date'] < (filtered_data['Order Date'].max() - pd.DateOffset(years=1))].groupby('Product Name')['Sales'].sum().max()
    top_product_rate = (filtered_data.groupby('Product Name')['Sales'].sum().loc[top_product] - prev_top_product_sales) / prev_top_product_sales * 100 if prev_top_product_sales else 0
    
    # Top Store (by State)
    top_store = filtered_data.groupby('State')['Sales'].sum().idxmax()
    prev_top_store_sales = filtered_data[filtered_data['Order Date'] < (filtered_data['Order Date'].max() - pd.DateOffset(years=1))].groupby('State')['Sales'].sum().max()
    top_store_rate = (filtered_data.groupby('State')['Sales'].sum().loc[top_store] - prev_top_store_sales) / prev_top_store_sales * 100 if prev_top_store_sales else 0
    
    # Average Sales
    avg_sales = filtered_data['Sales'].mean()
    prev_avg_sales = filtered_data[filtered_data['Order Date'] < (filtered_data['Order Date'].max() - pd.DateOffset(years=1))]['Sales'].mean()
    avg_sales_rate = (avg_sales - prev_avg_sales) / prev_avg_sales * 100 if prev_avg_sales else 0
    
    # Total Transactions
    total_transactions = filtered_data.shape[0]
    prev_total_transactions = filtered_data[filtered_data['Order Date'] < (filtered_data['Order Date'].max() - pd.DateOffset(years=1))].shape[0]
    total_transactions_rate = (total_transactions - prev_total_transactions) / prev_total_transactions * 100 if prev_total_transactions else 0
    
    # Unique Products
    unique_products = filtered_data['Product ID'].nunique()
    prev_unique_products = filtered_data[filtered_data['Order Date'] < (filtered_data['Order Date'].max() - pd.DateOffset(years=1))]['Product ID'].nunique()
    unique_products_rate = (unique_products - prev_unique_products) / prev_unique_products * 100 if prev_unique_products else 0

    # Creating 3 columns for the first row
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Sales", f"${total_sales:,.2f}", f"{total_sales_rate:.2f}%")

    with col2:
        st.metric("Top Product", top_product, f"{top_product_rate:.2f}%")

    with col3:
        st.metric("Top Store", top_store, f"{top_store_rate:.2f}%")

    # Creating 3 columns for the second row
    col4, col5, col6 = st.columns(3)

    with col4:
        st.metric("Average Sales", f"${avg_sales:,.2f}", f"{avg_sales_rate:.2f}%")

    with col5:
        st.metric("Total Transactions", f"{total_transactions}", f"{total_transactions_rate:.2f}%")

    with col6:
        st.metric("Unique Products", f"{unique_products}", f"{unique_products_rate:.2f}%")
else:
    col1.metric("Total Sales", "N/A")
    col2.metric("Top Product", "N/A")
    col3.metric("Top Store", "N/A")
    col4.metric("Average Sales", "N/A")
    col5.metric("Total Transactions", "0")
    col6.metric("Unique Products", "0")


# Visualizations
st.markdown('<h1 class="visualizations-header">Visualizations</h1>', unsafe_allow_html=True)


# Row B: Heatmap and Donut Chart
c1, c2 = st.columns((8, 2))

if not filtered_data.empty:
    with c1:
       # Initialize geolocator
        geolocator = Nominatim(user_agent="state_coordinates")

        # List of US states (you can replace this with your dataset)
        states = ["Alabama", "Alaska", "Arizona", "Arkansas", "California", "Colorado", "Connecticut", "Delaware", 
                "Florida", "Georgia", "Hawaii", "Idaho", "Illinois", "Indiana", "Iowa", "Kansas", "Kentucky", 
                "Louisiana", "Maine", "Maryland", "Massachusetts", "Michigan", "Minnesota", "Mississippi", "Missouri", 
                "Montana", "Nebraska", "Nevada", "New Hampshire", "New Jersey", "New Mexico", "New York", "North Carolina", 
                "North Dakota", "Ohio", "Oklahoma", "Oregon", "Pennsylvania", "Rhode Island", "South Carolina", 
                "South Dakota", "Tennessee", "Texas", "Utah", "Vermont", "Virginia", "Washington", "West Virginia", 
                "Wisconsin", "Wyoming"]

        # Create a dictionary to store state names and their corresponding coordinates
        state_coordinates = {}
        sales_data = []  # List to store sales data for each state

        # Get coordinates for each state and simulate sales data
        for state in states:
            location = geolocator.geocode(f"{state}, United States")
            if location:
                state_coordinates[state] = (location.latitude, location.longitude)
                sales_data.append(np.random.randint(100, 1000))  # Simulating sales data
            else:
                state_coordinates[state] = (None, None)
                sales_data.append(0)

        # Create a DataFrame with state coordinates and sales data
        df = pd.DataFrame(state_coordinates).T  # Transpose the dictionary to make state names as rows
        df.columns = ["lat", "lon"]
        df['Sales'] = sales_data

        # Displaying basic information
        st.markdown('### US Sales with Heatmap')
        st.markdown("This heatmap represents the sales data for each state with corresponding coordinates.")

        # Create a base map centered on the average coordinates
        m = folium.Map(location=[df['lat'].mean(), df['lon'].mean()], zoom_start=5)

        # Prepare data for heatmap (latitude, longitude, and sales data)
        heat_data = [[row['lat'], row['lon'], row['Sales']] for index, row in df.iterrows()]

        # Add HeatMap to the map
        HeatMap(heat_data).add_to(m)

        # Display the map in Streamlit
        st.components.v1.html(m._repr_html_(), width=700, height=500)

    with c2:
        st.markdown('#### Donut Chart: Sales Distribution by Store')
        # Ensure only relevant columns are aggregated
        donut_data = filtered_data.groupby('State', as_index=False)['Sales'].sum()
        pl.donut_chart(
            data=donut_data,
            theta='Sales',
            color='State',
            legend='bottom',
            use_container_width=True
        )
else:
    st.warning("No data available for visualizations based on the selected filters.")

# Row C: Line Chart
st.markdown('### Line Chart: Sales Over Time')
if not filtered_data.empty:
    line_chart_data = filtered_data.groupby('Order Date', as_index=False)['Sales'].sum()
    st.line_chart(line_chart_data.set_index('Order Date')['Sales'], height=300)
else:
    st.warning("No data available for the selected date range.")

# Row D: Scatter Plot
st.markdown("### Scatter Plot: Sales vs. Product Category and Customer Segment")
if not filtered_data.empty:
    scatter_chart = alt.Chart(filtered_data).mark_circle(size=60).encode(
        x='Sales:Q',  # Sales on the x-axis
        y='Category:N',  # Product Category on the y-axis
        color='Segment:N',  # Color by Customer Segment
        shape='Ship Mode:N',  # Shape by Ship Mode
        tooltip=['Order ID', 'Sales:Q', 'Category:N', 'Segment:N', 'Ship Mode:N']
    ).interactive()
    st.altair_chart(scatter_chart, use_container_width=True)
else:
    st.warning("No data available for scatter plot.")


st.markdown("### Box Plot: Sales by Ship Mode")
if not filtered_data.empty:
    box_chart = alt.Chart(filtered_data).mark_boxplot().encode(
        x='Ship Mode:N',
        y='Sales:Q',
        color='Ship Mode:N',
        tooltip=['Ship Mode', 'Sales:Q']
    ).interactive()
    st.altair_chart(box_chart, use_container_width=True)
else:
    st.warning("No data available for box plot.")

# Additional Insights
st.markdown('### Bar Chart: Sales by Store')
if not filtered_data.empty:
    bar_chart_data = filtered_data.groupby('State')['Sales'].sum()
    st.bar_chart(bar_chart_data)
else:
    st.warning("No data available for sales by store.")

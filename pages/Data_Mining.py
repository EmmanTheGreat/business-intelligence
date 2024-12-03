import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
#fix module

# Configure the page
st.set_page_config(layout='wide', initial_sidebar_state='expanded')

# Apply custom styles
try:
    with open('style.css') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
except FileNotFoundError:
    st.warning("Custom CSS file not found. Default styling applied.")


# Load and display dataset
@st.cache_data
def load_data():
    # Replace with your actual CSV file path
    df = pd.read_csv("clean_data.csv")
    return df

df = load_data()

# Display dataset in Streamlit
st.markdown("### Customer Data")
st.write(df.head(10))  # Display first few rows of the dataset

# Aggregate data by Customer ID
customer_data = df.groupby('Customer ID')['Sales'].sum().reset_index()

# Allow user to select the number of clusters
num_clusters = st.slider("Select the number of clusters", 2, 10, 3)

# Standardize the sales data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(customer_data[['Sales']])

# Apply K-Means clustering
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
customer_data['Cluster'] = kmeans.fit_predict(scaled_data)

# Displaying K-Means cluster centers
st.markdown("### K-Means Cluster Centers")
st.write(kmeans.cluster_centers_)

# Visualize the clusters using a scatter plot
st.markdown("### Customer Segmentation by Sales")
fig, ax = plt.subplots(figsize=(10, 6))

sns.scatterplot(data=customer_data, x=customer_data.index, y='Sales', hue='Cluster', palette='viridis', ax=ax)

# Adding labels and title
ax.set_xlabel('Customer Index')
ax.set_ylabel('Total Sales')
ax.set_title('Customer Segmentation by Sales')

# Display the plot in Streamlit
st.pyplot(fig)

# Show cluster details
st.markdown("### Cluster Details")
for cluster_num in range(num_clusters):
    st.write(f"**Cluster {cluster_num}:**")
    cluster = customer_data[customer_data['Cluster'] == cluster_num]
    st.write(cluster[['Customer ID','Sales']].head())

# Preprocess the dataset
df['Order Date'] = pd.to_datetime(df['Order Date'])  # Convert to datetime
df['Month'] = df['Order Date'].dt.month  # Extract the month
df['Year'] = df['Order Date'].dt.year  # Extract the year

# Aggregate sales by month and year
monthly_sales = df.groupby(['Year', 'Month'])['Sales'].sum().reset_index()

# Prepare data for regression
X = monthly_sales[['Year', 'Month']]
y = monthly_sales['Sales']

# Streamlit UI elements for interactivity
st.title("Sales Prediction with Linear Regression")

# User input for test size
test_size = st.slider('Select Test Size (%)', min_value=10, max_value=90, value=20, step=5) / 100

# User input for features (for example, you can add more features for a more advanced model)
features = st.multiselect('Select Features for Regression', ['Year', 'Month'], default=['Year', 'Month'])

# Recompute X based on selected features
X = monthly_sales[features]

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

# Train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, predictions)

# Display Mean Squared Error
st.write(f"Mean Squared Error: {mse:.2f}")

# Visualize the results
st.subheader("Sales Prediction Plot")
fig, ax = plt.subplots()
ax.scatter(range(len(y_test)), y_test, color='blue', label='Actual Sales')
ax.plot(range(len(predictions)), predictions, color='red', label='Predicted Sales')
ax.set_xlabel('Test Data Index')
ax.set_ylabel('Sales')
ax.set_title('Sales Prediction')
ax.legend()

# Display the plot in Streamlit
st.pyplot(fig)

# File uploader
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file:
    # Load the dataset
    df = pd.read_csv(uploaded_file)

    # Strip whitespace from column names
    df.columns = df.columns.str.strip()

    # Display dataset sample
    st.write("Sample of the uploaded dataset:")
    st.write(df.head())

    # Feature and target selection
    st.sidebar.header("Configuration")
    features = st.sidebar.multiselect(
        "Select Features for Prediction",
        options=df.columns,
        default=["Category", "Ship Mode", "Region"]  # Default feature columns
    )
    target = st.sidebar.selectbox(
        "Select Target Variable",
        options=df.columns,
        index=list(df.columns).index("Sales")  # Default target column
    )

    if features and target:
        # Prepare data for training
        X = df[features]
        y = df[target]

        # Filter for categorical features to apply OneHotEncoding
        categorical_features = [col for col in features if df[col].dtype == 'object']

        # Preprocessor for categorical features
        preprocessor = ColumnTransformer(
            transformers=[
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
            ],
            remainder='passthrough'  # Leave numerical features as is
        )

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Build the pipeline
        model = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
        ])

        # Train the model
        st.write("Training the model...")
        model.fit(X_train, y_train)

        # Make predictions
        predictions = model.predict(X_test)

        # Evaluate the model
        mse = mean_squared_error(y_test, predictions)
        st.write(f"Mean Squared Error (MSE): {mse:.2f}")

        # Visualization: Actual vs Predicted Sales
        st.subheader("Actual vs Predicted Sales")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(range(len(y_test)), y_test, color='blue', label='Actual Sales')
        ax.scatter(range(len(predictions)), predictions, color='red', label='Predicted Sales')
        ax.set_xlabel('Test Data Index')
        ax.set_ylabel('Sales')
        ax.set_title('Random Forest Sales Prediction')
        ax.legend()
        st.pyplot(fig)

        # Sidebar Filters for Features
        st.sidebar.header("Interactive Filters")
        filtered_data = df.copy()

        for feature in features:
            if df[feature].nunique() < 10:  # Display filter for categorical features
                selected_values = st.sidebar.multiselect(
                    f"Filter by {feature}",
                    options=df[feature].unique(),
                    default=df[feature].unique()
                )
                filtered_data = filtered_data[filtered_data[feature].isin(selected_values)]

        st.write("Filtered Data Preview:")
        st.write(filtered_data.head())

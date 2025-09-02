import streamlit as st
import pandas as pd
import requests
import json
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import time

# Page config
st.set_page_config(
    page_title="Diamond Price Predictor",
    page_icon="💎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #1f77b4;
    }
    .prediction-result {
        font-size: 2rem;
        font-weight: bold;
        color: #2e8b57;
        text-align: center;
        padding: 1rem;
        background-color: #f0fff0;
        border-radius: 0.5rem;
        border: 2px solid #2e8b57;
    }
</style>
""", unsafe_allow_html=True)

# API Configuration
API_BASE_URL = "http://api:5000"  # Docker service name

def call_prediction_api(data, endpoint="/api/v1/predict"):
    """Call the Flask API for predictions"""
    try:
        url = f"{API_BASE_URL}{endpoint}"
        response = requests.post(url, json=data, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError:
        st.error("❌ Cannot connect to API service. Please ensure the API container is running.")
        return None
    except requests.exceptions.Timeout:
        st.error("⏰ API request timed out. Please try again.")
        return None
    except requests.exceptions.RequestException as e:
        st.error(f"❌ API Error: {str(e)}")
        return None

def main():
    # Header
    st.markdown('<h1 class="main-header">💎 Diamond Price Predictor</h1>', unsafe_allow_html=True)
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choose a page", ["Single Prediction", "Batch Prediction", "Model Info", "Analytics"])
    
    if page == "Single Prediction":
        single_prediction_page()
    elif page == "Batch Prediction":
        batch_prediction_page()
    elif page == "Model Info":
        model_info_page()
    elif page == "Analytics":
        analytics_page()

def single_prediction_page():
    st.header("🔮 Single Diamond Price Prediction")
    
    # Create two columns for input
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Diamond Characteristics")
        carat = st.number_input("Carat Weight", min_value=0.1, max_value=10.0, value=1.0, step=0.1)
        cut = st.selectbox("Cut Quality", ["Fair", "Good", "Very Good", "Premium", "Ideal"])
        color = st.selectbox("Color Grade", ["D", "E", "F", "G", "H", "I", "J"])
        clarity = st.selectbox("Clarity", ["FL", "IF", "VVS1", "VVS2", "VS1", "VS2", "SI1", "SI2", "I1", "I2", "I3"])
    
    with col2:
        st.subheader("Physical Dimensions")
        depth = st.number_input("Depth %", min_value=40.0, max_value=80.0, value=61.5, step=0.1)
        table = st.number_input("Table %", min_value=40.0, max_value=80.0, value=57.0, step=0.1)
        x = st.number_input("Length (mm)", min_value=0.0, max_value=20.0, value=6.5, step=0.1)
        y = st.number_input("Width (mm)", min_value=0.0, max_value=20.0, value=6.5, step=0.1)
        z = st.number_input("Height (mm)", min_value=0.0, max_value=20.0, value=4.0, step=0.1)
    
    # Prediction button
    if st.button("💰 Predict Price", type="primary"):
        # Prepare data
        diamond_data = {
            "carat": carat,
            "cut": cut,
            "color": color,
            "clarity": clarity,
            "depth": depth,
            "table": table,
            "x": x,
            "y": y,
            "z": z
        }
        
        # Show loading spinner
        with st.spinner("Predicting diamond price..."):
            result = call_prediction_api(diamond_data)
        
        if result and result.get("success"):
            # Display prediction result
            predicted_price = result.get("predicted_price", 0)
            st.markdown(f'<div class="prediction-result">Predicted Price: ${predicted_price:,.2f}</div>', 
                       unsafe_allow_html=True)
            
            # Show prediction confidence if available
            if "confidence" in result:
                st.info(f"🎯 Prediction Confidence: {result['confidence']:.1%}")
            
            # Show diamond summary
            st.subheader("Diamond Summary")
            summary_cols = st.columns(3)
            with summary_cols[0]:
                st.metric("Carat", f"{carat:.2f}")
                st.metric("Cut", cut)
                st.metric("Color", color)
            with summary_cols[1]:
                st.metric("Clarity", clarity)
                st.metric("Depth", f"{depth:.1f}%")
                st.metric("Table", f"{table:.1f}%")
            with summary_cols[2]:
                st.metric("Length", f"{x:.1f}mm")
                st.metric("Width", f"{y:.1f}mm")
                st.metric("Height", f"{z:.1f}mm")

def batch_prediction_page():
    st.header("📊 Batch Diamond Price Prediction")
    
    st.info("Upload a CSV file with diamond characteristics to get predictions for multiple diamonds.")
    
    # File uploader
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            # Read CSV
            df = pd.read_csv(uploaded_file)
            
            st.subheader("📋 Uploaded Data Preview")
            st.dataframe(df.head(), use_container_width=True)
            
            st.subheader("📈 Data Summary")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Diamonds", len(df))
            with col2:
                st.metric("Columns", len(df.columns))
            with col3:
                if 'carat' in df.columns:
                    st.metric("Avg Carat", f"{df['carat'].mean():.2f}")
            
            # Prediction button
            if st.button("🔮 Predict All Prices", type="primary"):
                with st.spinner("Processing batch predictions..."):
                    # Convert DataFrame to list of dictionaries
                    batch_data = df.to_dict('records')
                    result = call_prediction_api({"diamonds": batch_data}, "/api/v1/predict/batch")
                
                if result and result.get("success"):
                    predictions = result.get("predictions", [])
                    
                    # Add predictions to DataFrame
                    df['predicted_price'] = [p.get("predicted_price", 0) for p in predictions]
                    
                    st.subheader("🎯 Prediction Results")
                    st.dataframe(df, use_container_width=True)
                    
                    # Download button for results
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Results",
                        data=csv,
                        file_name=f"diamond_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                    
                    # Show statistics
                    st.subheader("📊 Batch Statistics")
                    stats_cols = st.columns(4)
                    with stats_cols[0]:
                        st.metric("Total Predictions", len(predictions))
                    with stats_cols[1]:
                        st.metric("Avg Predicted Price", f"${df['predicted_price'].mean():,.2f}")
                    with stats_cols[2]:
                        st.metric("Min Price", f"${df['predicted_price'].min():,.2f}")
                    with stats_cols[3]:
                        st.metric("Max Price", f"${df['predicted_price'].max():,.2f}")
                    
                    # Price distribution chart
                    fig = px.histogram(df, x='predicted_price', nbins=30, 
                                     title="Price Distribution")
                    fig.update_layout(xaxis_title="Predicted Price ($)", yaxis_title="Count")
                    st.plotly_chart(fig, use_container_width=True)
                    
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")

def model_info_page():
    st.header("🤖 Model Information")
    
    # Try to get model info from API
    try:
        response = requests.get(f"{API_BASE_URL}/api/v1/model/info", timeout=5)
        if response.status_code == 200:
            model_info = response.json()
            
            st.subheader("Model Details")
            info_cols = st.columns(2)
            
            with info_cols[0]:
                st.info(f"**Model Type:** {model_info.get('model_type', 'XGBoost')}")
                st.info(f"**Version:** {model_info.get('version', '1.0.0')}")
                st.info(f"**Training Date:** {model_info.get('training_date', 'Unknown')}")
            
            with info_cols[1]:
                st.info(f"**Accuracy:** {model_info.get('accuracy', 'N/A')}")
                st.info(f"**Features:** {model_info.get('features', 9)}")
                st.info(f"**Dataset Size:** {model_info.get('training_samples', 'N/A')}")
                
        else:
            st.warning("⚠️ Could not retrieve model information from API")
            
    except requests.exceptions.ConnectionError:
        st.error("❌ Cannot connect to API service")
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
    
    # Static model information
    st.subheader("📋 Model Features")
    features_df = pd.DataFrame({
        'Feature': ['carat', 'cut', 'color', 'clarity', 'depth', 'table', 'x', 'y', 'z'],
        'Type': ['Numeric', 'Categorical', 'Categorical', 'Categorical', 'Numeric', 'Numeric', 'Numeric', 'Numeric', 'Numeric'],
        'Description': [
            'Diamond weight in carats',
            'Cut quality (Fair, Good, Very Good, Premium, Ideal)',
            'Color grade (D-J scale)',
            'Clarity grade (FL to I3)',
            'Depth percentage',
            'Table percentage', 
            'Length in mm',
            'Width in mm',
            'Height in mm'
        ]
    })
    st.dataframe(features_df, use_container_width=True)

def analytics_page():
    st.header("📈 Analytics Dashboard")
    
    st.info("🚧 Analytics features coming soon! This will include:")
    st.write("""
    - **Prediction History**: Track all predictions over time
    - **Price Trends**: Analyze diamond price patterns
    - **Feature Importance**: Understand which characteristics most affect price
    - **Model Performance**: Real-time accuracy metrics
    - **Usage Statistics**: API usage and user behavior analytics
    """)
    
    # Placeholder charts
    st.subheader("Sample Analytics")
    
    # Create sample data for demonstration
    dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
    num_days = len(dates)
    sample_data = pd.DataFrame({
        'date': dates,
        'predictions': list(range(num_days)),
        'avg_price': [5000 + i + (i % 100 * 50) for i in range(num_days)]
    })
    
    # Line chart
    fig = px.line(sample_data.iloc[::30], x='date', y='predictions', 
                  title="Daily Prediction Volume (Sample Data)")
    st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
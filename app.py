import streamlit as st
import pandas as pd
import joblib
import datetime
import plotly.graph_objects as go

# --- 1. Page Configuration & Setup ---
st.set_page_config(
    page_title="Smart Grid Load Forecaster",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- 2. Custom CSS for a Stunning UI ---
st.markdown("""
    <style>
    /* Main background */
    .stApp {
        background-color: #0e1117;
    }
    /* Title styling */
    h1 {
        color: #00d4ff;
        text-align: center;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        font-weight: 800;
        letter-spacing: 1px;
    }
    /* Button styling */
    .stButton>button {
        background-color: #00d4ff;
        color: #0e1117;
        font-weight: 900;
        font-size: 1.1rem;
        border-radius: 10px;
        width: 100%;
        padding: 15px;
        border: None;
        transition: all 0.4s ease;
        box-shadow: 0 4px 15px rgba(0, 212, 255, 0.3);
    }
    .stButton>button:hover {
        background-color: #ffffff;
        color: #000000;
        box-shadow: 0 6px 20px rgba(0, 212, 255, 0.6);
        transform: translateY(-2px);
    }
    /* Prediction container */
    .prediction-container {
        background: linear-gradient(145deg, #1e2129, #15181e);
        padding: 20px;
        border-radius: 20px;
        box-shadow:  8px 8px 16px #0b0d12, -8px -8px 16px #11151c;
        margin-top: 30px;
    }
    </style>
""", unsafe_allow_html=True)

# --- 3. Header Section ---
st.markdown("<h1>⚡ Smart Grid AI Load Forecaster</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #8b949e; font-size: 1.2rem; margin-bottom: 40px;'>University Graduation Project | Advanced Machine Learning Prediction Engine</p>", unsafe_allow_html=True)

# --- 4. Model Loading ---
@st.cache_resource
def load_model():
    return joblib.load('xgboost_smartgrid_model.pkl')

try:
    model = load_model()
except Exception as e:
    st.error(f"Model File Error: {e}")
    st.stop()

# --- 5. User Input Interface (3 Columns Layout) ---
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### 🕒 Temporal Context")
    input_time = st.time_input("Target Time", datetime.time(20, 0), help="Select the exact time for the prediction.")
    day_of_week = st.selectbox(
        "Day of the Week", 
        options=[0, 1, 2, 3, 4, 5, 6], 
        format_func=lambda x: ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"][x]
    )
    is_weekend = st.radio("Is it a Weekend?", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No", horizontal=True)

with col2:
    st.markdown("### 🔌 Grid Telemetry")
    avg_voltage = st.number_input(
        "Average Voltage (V)", 
        min_value=200.0, max_value=250.0, value=230.0, step=0.1, 
        help="Current simulated voltage in the grid."
    )

with col3:
    st.markdown("### 📊 Historical Load")
    lag_1 = st.number_input("Load 5 mins ago (kW)", min_value=0.0, max_value=20.0, value=3.5, step=0.1)
    lag_2 = st.number_input("Load 10 mins ago (kW)", min_value=0.0, max_value=20.0, value=3.4, step=0.1)
    rolling_mean = st.number_input("15-min Rolling Mean (kW)", min_value=0.0, max_value=20.0, value=3.45, step=0.1)

st.markdown("<br>", unsafe_allow_html=True)

# --- 6. Prediction Execution & Visualization ---
if st.button("🔮 INITIALIZE AI PREDICTION"):
    
    # Prepare Data mapping
    raw_data = {
        'avg_voltage': [avg_voltage],
        'minute': [input_time.minute],
        'hour': [input_time.hour],
        'dayofweek': [day_of_week],
        'is_weekend': [is_weekend],
        'lag_1_power': [lag_1],
        'lag_2_power': [lag_2],
        'rolling_mean_3': [rolling_mean]
    }
    
    input_df = pd.DataFrame(raw_data)
    
    # Enforce strict column order based on training phase
    correct_feature_order = [
        'avg_voltage', 'minute', 'hour', 'dayofweek', 
        'is_weekend', 'lag_1_power', 'lag_2_power', 'rolling_mean_3'
    ]
    input_df = input_df[correct_feature_order]
    
    # Execute prediction with a loading spinner
    with st.spinner("Analyzing neural patterns and grid historical data..."):
        try:
            prediction = model.predict(input_df)[0]
            
            # Create a stunning Gauge Chart using Plotly
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = prediction,
                title = {'text': "Predicted Required Load (kW)", 'font': {'size': 26, 'color': '#8b949e'}},
                number = {'font': {'size': 60, 'color': '#00d4ff'}, 'valueformat': '.3f'},
                gauge = {
                    'axis': {'range': [0, 10], 'tickwidth': 1, 'tickcolor': "white"},
                    'bar': {'color': "#00d4ff", 'thickness': 0.25},
                    'bgcolor': "rgba(0,0,0,0)",
                    'borderwidth': 0,
                    'steps': [
                        {'range': [0, 4], 'color': "rgba(39, 174, 96, 0.2)"},   # Safe / Low Load
                        {'range': [4, 7], 'color': "rgba(241, 196, 15, 0.2)"},  # Medium Load
                        {'range': [7, 10], 'color': "rgba(231, 76, 60, 0.2)"}   # High / Peak Load
                    ],
                }
            ))
            
            # Apply transparent background to the chart
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font={'color': "white"}, height=450, margin=dict(t=50, b=0, l=0, r=0))
            
            # Display chart inside the custom CSS container
            st.markdown("<div class='prediction-container'>", unsafe_allow_html=True)
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
            
            st.success("✅ AI prediction successful. System is ready to allocate power accordingly.")
            
        except Exception as e:
            st.error(f"System Error: {e}")

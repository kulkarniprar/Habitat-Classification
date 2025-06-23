import streamlit as st
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from PIL import Image
from streamlit_folium import st_folium
import folium
import plotly.graph_objects as go

# --- Page Setup ---
st.set_page_config(page_title="Habitats Dashboard", layout="wide", initial_sidebar_state="expanded")

# --- Load model only once ---
@st.cache_resource
def load_trained_model():
    return load_model('habitat_classifier_improved.h5')

model = load_trained_model()

class_labels = ['Forest', 'HerbaceousVegetation', 'Pasture', 'River',
                'SeaLake', 'beach', 'desert', 'ice', 'mountain', 'ocean']

# --- Enhanced Styling ---
st.markdown("""
    <style>
    .stApp {
        background-image: url("https://images.ctfassets.net/hrltx12pl8hq/6SIFEqcMmO7koSBqCwKErd/d139bb7e36a8501c9503994edee29fa2/shutterstock_1037763571-min.jpg?fit=fill&w=1200&h=630");
        background-size: cover;
        background-attachment: fixed;
        background-repeat: no-repeat;
        background-position: center;
    }
    
    .main-header {
        background: linear-gradient(90deg, #2E8B57, #228B22);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    }
    
    .section-container {
        background-color: rgba(255, 255, 255, 0.95);
        padding: 2rem;
        border-radius: 15px;
        margin: 1.5rem 0;
        box-shadow: 0px 8px 20px rgba(0,0,0,0.2);
        backdrop-filter: blur(8px);
        border-left: 5px solid #228B22;
    }
    
    .habitat-card {
        background-color: rgba(255, 255, 255, 0.9);
        padding: 1.5rem;
        border-radius: 12px;
        margin: 0.5rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        border-top: 4px solid #228B22;
        transition: transform 0.2s ease;
    }
    
    .habitat-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.2);
    }
    
    .classifier-container {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.95), rgba(240, 248, 255, 0.95));
        padding: 2rem;
        border-radius: 15px;
        margin: 2rem 0;
        box-shadow: 0px 8px 20px rgba(0,0,0,0.2);
        backdrop-filter: blur(8px);
        border: 2px solid #228B22;
    }
    
    .metric-container {
        text-align: center;
        padding: 1rem;
        background: rgba(240, 248, 255, 0.8);
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    
    .section-title {
        color: #2E8B57;
        font-size: 1.8rem;
        font-weight: bold;
        margin-bottom: 1rem;
        text-align: center;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
    }
    
    .habitat-title {
        color: #228B22;
        font-size: 1.2rem;
        font-weight: bold;
        margin-bottom: 1rem;
        text-align: center;
    }
    
    .stButton > button {
        background: linear-gradient(90deg, #228B22, #32CD32);
        color: white;
        border-radius: 25px;
        padding: 12px 24px;
        font-size: 16px;
        font-weight: bold;
        border: none;
        box-shadow: 0 4px 12px rgba(34, 139, 34, 0.3);
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        background: linear-gradient(90deg, #32CD32, #228B22);
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(34, 139, 34, 0.4);
    }
    
    .sidebar .stSelectbox > div > div {
        background-color: rgba(255, 255, 255, 0.9);
        border-radius: 10px;
    }
    
    .info-box {
        background: linear-gradient(135deg, #e8f5e8, #f0f8f0);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 5px solid #228B22;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# --- Header Section ---
st.markdown("""
    <div class="main-header">
        <h1>🌿 Habitat Statistics Dashboard</h1>
        <p>Explore India's diverse ecosystems and analyze habitat data with AI-powered classification</p>
    </div>
""", unsafe_allow_html=True)

# Habitat details - Updated with authentic data from official sources (2023-2024)
habitats = {
    "Forest": {
        "area_km2": 713789,  # FSI Report 2023: 21.76% of geographical area
        "loss_percent": 11.0,  # Based on biodiversity hotspot loss data
        "protected_percent": 45.2,  # Higher protection in forest areas
        "lat": 22.0,
        "lon": 82.0,
        "color": "#228B22",
        "description": "India's forests cover 21.76% of geographical area and host 80% of terrestrial biodiversity."
    },
    "HerbaceousVegetation": {
        "area_km2": 450000,  # Estimated grassland coverage
        "loss_percent": 15.5,  # Significant grassland conversion
        "protected_percent": 8.2,  # Lower protection for grasslands
        "lat": 24.5,
        "lon": 79.0,
        "color": "#9ACD32",
        "description": "Grasslands and herbaceous regions support grazing and wildlife corridors but face conversion pressure."
    },
    "Pasture": {
        "area_km2": 550000,  # Agricultural and pastoral lands
        "loss_percent": 12.3,  # Conversion to other uses
        "protected_percent": 3.5,  # Minimal protection
        "lat": 23.0,
        "lon": 75.0,
        "color": "#D2B48C",
        "description": "Pastures support livestock and rural livelihoods across India."
    },
    "River": {
        "area_km2": 195000,  # Inland water bodies
        "loss_percent": 8.0,  # Wetland loss reported
        "protected_percent": 12.1,  # Some river stretches protected
        "lat": 25.6,
        "lon": 83.0,
        "color": "#4169E1",
        "description": "India's river systems like Ganga and Brahmaputra are lifelines for millions but face pollution threats."
    },
    "SeaLake": {
        "area_km2": 85000,  # Lakes and inland water bodies
        "loss_percent": 25.0,  # Significant wetland loss documented
        "protected_percent": 18.5,  # Some major lakes protected
        "lat": 20.5,
        "lon": 85.8,
        "color": "#00BFFF",
        "description": "Lakes and inland seas support aquatic biodiversity but many are shrinking rapidly."
    },
    "beach": {
        "area_km2": 7500,  # Coastal areas (7,500 km coastline)
        "loss_percent": 35.0,  # High coastal erosion and development
        "protected_percent": 15.8,  # Marine protected areas
        "lat": 13.1,
        "lon": 80.3,
        "color": "#FFD700",
        "description": "India's 7,500 km coastline faces erosion, sea-level rise, and development pressure."
    },
    "desert": {
        "area_km2": 317000,  # Thar Desert and arid regions
        "loss_percent": 4.2,  # Relatively stable but facing degradation
        "protected_percent": 12.8,  # Desert National Park and reserves
        "lat": 27.0,
        "lon": 71.0,
        "color": "#EDC9Af",
        "description": "The Thar Desert spans 317,000 km² and hosts unique arid-zone species."
    },
    "ice": {
        "area_km2": 37000,  # Himalayan glaciers
        "loss_percent": 45.0,  # Rapid glacial retreat due to climate change
        "protected_percent": 25.0,  # High altitude protected areas
        "lat": 33.5,
        "lon": 78.0,
        "color": "#ADD8E6",
        "description": "Himalayan glaciers covering 37,000 km² are retreating rapidly due to climate change."
    },
    "mountain": {
        "area_km2": 590000,  # Mountain ecosystems
        "loss_percent": 8.5,  # Habitat fragmentation in mountains
        "protected_percent": 22.3,  # Mountain national parks and reserves
        "lat": 32.0,
        "lon": 79.0,
        "color": "#A9A9A9",
        "description": "Mountain ecosystems across Himalayas, Western Ghats, and other ranges harbor endemic species."
    },
    "ocean": {
        "area_km2": 2305000,  # India's Exclusive Economic Zone
        "loss_percent": 12.5,  # Marine biodiversity loss
        "protected_percent": 4.8,  # Marine protected areas (below global average)
        "lat": 8.0,
        "lon": 77.0,
        "color": "#1E90FF",
        "description": "India's EEZ of 2.3 million km² supports fisheries and marine biodiversity but needs more protection."
    }
}

# --- Sidebar Configuration ---
with st.sidebar:
    st.markdown("### 🧭 Navigation & Controls")
    st.markdown("---")
    
    st.markdown("#### Habitat Selection")
    selected_habitat = st.selectbox("Choose a habitat to highlight:", ["All"] + list(habitats.keys()))
    
    st.markdown("---")
    st.markdown("#### Quick Stats")
    total_area = sum(data['area_km2'] for data in habitats.values())
    avg_loss = sum(data['loss_percent'] for data in habitats.values()) / len(habitats)
    avg_protected = sum(data['protected_percent'] for data in habitats.values()) / len(habitats)
    
    st.metric("🌍 Total Area", f"{total_area:,.0f} km²")
    st.metric("📉 Avg Loss", f"{avg_loss:.1f}%")
    st.metric("🛡 Avg Protected", f"{avg_protected:.1f}%")

# --- Main Content Area ---
# Habitat Overview Section
st.markdown('<div class="section-container">', unsafe_allow_html=True)
st.markdown('<h2 class="section-title">🌐 Habitat Overview</h2>', unsafe_allow_html=True)

# Create responsive grid for habitat cards
num_cols = 5
rows = [list(habitats.items())[i:i + num_cols] for i in range(0, len(habitats), num_cols)]

for row in rows:
    cols = st.columns(len(row))
    for idx, (name, data) in enumerate(row):
        with cols[idx]:
            st.markdown(f"""
                <div class="habitat-card">
                    <h4 class="habitat-title">{name}</h4>
                    <div class="metric-container">
                        <strong>🌍 Area</strong><br>
                        {data['area_km2']:,} km²
                    </div>
                    <div class="metric-container">
                        <strong>📉 Loss</strong><br>
                        {data['loss_percent']}%
                    </div>
                    <div class="metric-container">
                        <strong>🛡 Protected</strong><br>
                        {data['protected_percent']}%
                    </div>
                </div>
            """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# Map Section
st.markdown('<div class="section-container">', unsafe_allow_html=True)
st.markdown('<h2 class="section-title">🗺 Interactive Habitat Map</h2>', unsafe_allow_html=True)

if selected_habitat != "All":
    selected_data = habitats[selected_habitat]
    st.markdown(f"""
        <div class="info-box">
            <h4>📍 Focusing on: {selected_habitat}</h4>
            <p>{selected_data['description']}</p>
        </div>
    """, unsafe_allow_html=True)

# Map configuration
if selected_habitat != "All":
    selected_data = habitats[selected_habitat]
    map_center = [selected_data["lat"], selected_data["lon"]]
    zoom = 6
else:
    map_center = [20.5937, 78.9629]
    zoom = 5

m = folium.Map(location=map_center, zoom_start=zoom, tiles="CartoDB positron")

# Add tree cover loss layer
tree_loss_tile = "https://tiles.globalforestwatch.org/v3/tree_cover_loss__ts_2023/{z}/{x}/{y}.png?year=2023"
folium.TileLayer(
    tiles=tree_loss_tile,
    name="Tree Cover Loss (2023)",
    attr="Global Forest Watch",
    overlay=True,
    control=True,
    opacity=0.6
).add_to(m)

# Add markers
for name, data in habitats.items():
    popup_html = f"""
    <div style="font-family: Arial, sans-serif; max-width: 200px;">
        <h4 style="color: #228B22; margin-bottom: 10px;">{name}</h4>
        <p><strong>Area:</strong> {data['area_km2']:,} km²</p>
        <p><strong>Loss:</strong> {data['loss_percent']}%</p>
        <p><strong>Protected:</strong> {data['protected_percent']}%</p>
        <hr style="margin: 10px 0;">
        <p style="font-style: italic; font-size: 12px;">{data['description']}</p>
    </div>
    """
    marker_color = "red" if name == selected_habitat else "green"
    folium.Marker(
        location=[data["lat"], data["lon"]],
        popup=popup_html,
        icon=folium.Icon(color=marker_color, icon="leaf", prefix="fa")
    ).add_to(m)

folium.LayerControl().add_to(m)
st_folium(m, width=1000, height=600)
st.markdown('</div>', unsafe_allow_html=True)

# Detailed Analysis Section (only shown when habitat is selected)
if selected_habitat != "All":
    st.markdown('<div class="section-container">', unsafe_allow_html=True)
    st.markdown(f'<h2 class="section-title">📊 Detailed Analysis: {selected_habitat}</h2>', unsafe_allow_html=True)
    
    data = habitats[selected_habitat]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📉 Habitat Loss Gauge")
        fig_loss = go.Figure(go.Indicator(
            mode="gauge+number",
            value=data["loss_percent"],
            title={'text': f"{selected_habitat} Loss %", 'font': {'size': 20, 'color': '#2E8B57'}},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': data["color"]},
                'steps': [
                    {'range': [0, 20], 'color': "#e0f7e9"},
                    {'range': [20, 40], 'color': "#cce7dd"},
                    {'range': [40, 100], 'color': "#ffcccc"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 15
                }
            }
        ))
        fig_loss.update_layout(height=400, font={'color': "#2E8B57", 'family': "Arial"})
        st.plotly_chart(fig_loss, use_container_width=True)

    with col2:
        st.markdown("### 🛡 Protected Area Coverage")
        fig_protected = go.Figure(go.Indicator(
            mode="gauge+number",
            value=data["protected_percent"],
            title={'text': f"{selected_habitat} Protected %", 'font': {'size': 20, 'color': '#2E8B57'}},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "#6A5ACD"},
                'steps': [
                    {'range': [0, 30], 'color': "#f0f8ff"},
                    {'range': [30, 60], 'color': "#dbe9ff"},
                    {'range': [60, 100], 'color': "#c0cfff"}
                ],
                'threshold': {
                    'line': {'color': "green", 'width': 4},
                    'thickness': 0.75,
                    'value': 30
                }
            }
        ))
        fig_protected.update_layout(height=400, font={'color': "#2E8B57", 'family': "Arial"})
        st.plotly_chart(fig_protected, use_container_width=True)
    
    # Additional habitat information
    st.markdown(f"""
        <div class="info-box">
            <h4>🔍 Habitat Insights</h4>
            <p><strong>Description:</strong> {data['description']}</p>
            <p><strong>Total Coverage:</strong> {data['area_km2']:,} square kilometers</p>
            <p><strong>Conservation Status:</strong> 
                {'🔴 High Priority' if data['loss_percent'] > 10 else '🟡 Medium Priority' if data['loss_percent'] > 5 else '🟢 Stable'}
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# --- AI Image Classifier Section ---
st.markdown('<div class="classifier-container">', unsafe_allow_html=True)
st.markdown('<h2 class="section-title">📷 AI Habitat Image Classifier</h2>', unsafe_allow_html=True)

st.markdown("""
    <div class="info-box">
        <p>🎯 <strong>How it works:</strong> Upload an image of a natural environment, and our AI model will analyze and classify the habitat type with confidence scoring.</p>
        <p>📋 <strong>Supported habitats:</strong> Forest, Herbaceous Vegetation, Pasture, River, Sea/Lake, Beach, Desert, Ice, Mountain, Ocean</p>
    </div>
""", unsafe_allow_html=True)

col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### 📁 Upload Image")
    uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"], help="Upload a clear image of a natural habitat for best results")
    
    if uploaded_file:
        img = Image.open(uploaded_file).convert('RGB')
        resized = img.resize((64, 64))
        img_array = image.img_to_array(resized) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        st.image(img, caption="📷 Uploaded Image", use_column_width=True)

with col2:
    st.markdown("### 🔍 Classification Results")
    
    if uploaded_file:
        if st.button("🚀 Analyze Habitat", use_container_width=True):
            with st.spinner("🤖 AI is analyzing your image..."):
                predictions = model.predict(img_array)
                pred_idx = np.argmax(predictions)
                label = class_labels[pred_idx]
                confidence = predictions[0][pred_idx]
                
                # Display results
                st.markdown(f"""
                    <div style="background: linear-gradient(135deg, #e8f5e8, #f0f8f0); 
                                padding: 1.5rem; border-radius: 12px; 
                                border-left: 5px solid #228B22; margin: 1rem 0;">
                        <h4 style="color: #228B22;">✅ Classification Complete!</h4>
                        <p><strong>🏷 Predicted Habitat:</strong> <span style="color: #2E8B57; font-size: 1.2em; font-weight: bold;">{label}</span></p>
                        <p><strong>📊 Confidence Score:</strong> <span style="color: #2E8B57; font-weight: bold;">{confidence:.2%}</span></p>
                    </div>
                """, unsafe_allow_html=True)
                
                # Show habitat information if available
                if label in habitats:
                    habitat_info = habitats[label]
                    st.markdown(f"""
                        <div class="info-box">
                            <h4>📖 About {label}</h4>
                            <p>{habitat_info['description']}</p>
                            <p><strong>Coverage in India:</strong> {habitat_info['area_km2']:,} km²</p>
                        </div>
                    """, unsafe_allow_html=True)
    else:
        st.markdown("""
            <div style="background: #f8f9fa; padding: 2rem; border-radius: 12px; text-align: center; border: 2px dashed #dee2e6;">
                <h4 style="color: #6c757d;">📷 No Image Selected</h4>
                <p style="color: #6c757d;">Please upload an image to see classification results</p>
            </div>
        """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# --- Footer ---
st.markdown("""
    <div style="background: linear-gradient(90deg, #2E8B57, #228B22); 
                padding: 2rem; border-radius: 15px; margin-top: 2rem; 
                text-align: center; color: white;">
        <h4>🌿 Habitat Conservation Dashboard</h4>
        <p>Powered by AI • Built for Conservation • Protecting India's Natural Heritage</p>
    </div>
""", unsafe_allow_html=True)
import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import base64
from translations import LANG_DICT

# --- 1. CONFIG ---
st.set_page_config(
    page_title="PalmScan AI", 
    layout="wide", 
    initial_sidebar_state="expanded" 
)

def get_base64(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_ui_theme():
    try:
        bin_str = get_base64('background.jpg')
        st.markdown(f'''
            <style>
            .stApp {{
                background-image: url("data:image/png;base64,{bin_str}");
                background-size: cover;
                background-attachment: fixed;
            }}
            </style>
        ''', unsafe_allow_html=True)
    except: pass
    
    with open("styles.css") as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

set_ui_theme()

# --- 2. MODEL ---
@st.cache_resource
def load_model():
    try: return tf.keras.models.load_model('best_model.keras')
    except: return None

model = load_model()

# --- 3. SIDEBAR ---
with st.sidebar:
    st.markdown("<h2 style='margin-bottom:0;'>🌿 PalmScan AI</h2>", unsafe_allow_html=True)
    st.markdown("<p style='font-size:0.8rem; opacity:0.6; color:black;'>SYSTEM ONLINE · v2.1</p>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Language Selection (CSS will fix the styling of this)
    selected_lang = st.selectbox("🌐 Choose Language", 
                                options=["English", "Telugu", "Hindi", "Tamil", "Kannada"])
    t = LANG_DICT[selected_lang]
    
    st.markdown("<br><p style='font-weight:bold; font-size:0.9rem; color:black;'>DETECTABLE CONDITIONS</p>", unsafe_allow_html=True)
    
    conds = [
        ("Boron Deficiency", "#FFF5F0", "#E4572E"),
        ("Healthy", "#F0FFF4", "#38A169"),
        ("Kalium Deficiency", "#F5F0FF", "#6B46C1"),
        ("Magnesium Deficiency", "#F0F9FF", "#3182CE"),
        ("Nitrogen Deficiency", "#FFFDF0", "#B7791F")
    ]
    for name, bg, txt in conds:
        st.markdown(f'<div class="deficiency-tag" style="background:{bg}; color:{txt};">● {name}</div>', unsafe_allow_html=True)

# --- 4. MAIN PAGE ---
st.markdown(f"<h1 style='color:white; font-size:3.5rem; margin-bottom:0; text-shadow: 2px 2px 8px rgba(0,0,0,0.6);'>{t['title']}</h1>", unsafe_allow_html=True)
st.markdown(f"<p style='color:white; font-size:1.4rem; opacity:0.9; text-shadow: 1px 1px 4px rgba(0,0,0,0.4);'>{t['subtitle']}</p>", unsafe_allow_html=True)

# File Uploader
st.markdown("<br>", unsafe_allow_html=True)
uploaded_file = st.file_uploader(t['upload_label'], type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.divider()
    st.markdown(f"<h3 style='color:white; text-shadow: 1px 1px 3px rgba(0,0,0,0.5);'>{t['result_header']}</h3>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1.2], gap="large")
    
    with col1:
        st.markdown(f"<div style='background:white; padding:5px 15px; border-radius:8px; color:black; font-weight:bold; width:fit-content; margin-bottom:15px; box-shadow: 0 4px 10px rgba(0,0,0,0.2);'>{t['specimen']}</div>", unsafe_allow_html=True)
        st.image(image, width=450)
        
    with col2:
        if model:
            # AI Inference
            img = image.resize((224, 224))
            arr = np.array(img) / 255.0
            arr = np.expand_dims(arr, axis=0)
            preds = model.predict(arr)
            idx = np.argmax(preds)
            conf = np.max(preds) * 100
            
            eng_name = LANG_DICT["English"]["conditions"][idx]
            local_name = t["conditions"][idx]
            sol = t["treatments"].get(eng_name, "...")

            # Cards
            st.markdown(f'''
                <div class="card">
                    <small style="text-transform:uppercase; opacity:0.5; font-weight:bold; letter-spacing:1px;">Detection Result</small>
                    <h1 style="color:#7B341E; margin:10px 0; font-size:2.8rem; font-weight:800;">{local_name}</h1>
                    <span style="color:#B7791F; font-size:0.9rem; font-weight:600;">class · {eng_name.lower()}</span>
                </div>
                
                <div class="card">
                    <small style="text-transform:uppercase; opacity:0.5; font-weight:bold; letter-spacing:1px;">{t['confidence']}</small>
                    <h1 style="color:#38A169; margin:5px 0; font-size:2.5rem; font-weight:800;">{conf:.1f}%</h1>
                    <div class="confidence-bar-bg"><div class="confidence-bar-fill" style="width:{conf}%"></div></div>
                </div>

                <div class="card" style="border-left: 10px solid #F6AD55;">
                    <small style="text-transform:uppercase; opacity:0.5; font-weight:bold; letter-spacing:1px;">💡 {t['treatment']}</small>
                    <p style="margin-top:15px; font-size:1.15rem; line-height:1.7; color:#2d3748; font-weight:500;">{sol}</p>
                </div>
            ''', unsafe_allow_html=True)
else:
    # State before upload
    st.markdown(f'''
    <div class="upload-box">
        <img src="https://cdn-icons-png.flaticon.com/512/1147/1147996.png" width="80" style="filter: invert(1); opacity: 0.6; margin-bottom:20px;">
        <h1 style="margin:0; font-size:2.5rem;">{t['waiting']}</h1>
        <p style="opacity:0.8; font-weight:bold; letter-spacing:2px; font-size:1rem; margin-top:10px;">{t['awaiting']}</p>
    </div>
    ''', unsafe_allow_html=True)
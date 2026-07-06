import os
import numpy as np
import pandas as pd
import joblib
import streamlit as st
import shap
import matplotlib.pyplot as plt

st.set_page_config(page_title='Credit Card Fraud Detector', page_icon='💳', layout='wide')


@st.cache_resource
def load_model():
    return joblib.load(os.path.join('models', 'rf.pkl'))


@st.cache_resource
def load_scaler():
    scaler_path = os.path.join('models', 'scaler.pkl')
    if os.path.exists(scaler_path):
        return joblib.load(scaler_path)
    st.warning('Scaler not found. Run `python src/save_scaler.py` first.')
    return None


@st.cache_resource
def get_explainer(_model):
    return shap.TreeExplainer(_model)


def main():
    st.title('💳 Credit Card Fraud Detection')
    
    model = load_model()
    scaler = load_scaler()
    explainer = get_explainer(model)
    
    if scaler is None:
        st.error('Run `python src/save_scaler.py` to save the scaler first.')
        st.stop()
    
    # Sidebar inputs
    st.sidebar.header('Transaction Features')
    
    amount = st.sidebar.number_input('Amount (€)', 0.0, 10000.0, 100.0, 10.0)
    time = st.sidebar.number_input('Time (seconds)', 0.0, 172800.0, 50000.0, 1000.0)
    
    st.sidebar.markdown('### PCA Features V1–V5')
    v1 = st.sidebar.slider('V1', -5.0, 5.0, 0.0, 0.1)
    v2 = st.sidebar.slider('V2', -5.0, 5.0, 0.0, 0.1)
    v3 = st.sidebar.slider('V3', -5.0, 5.0, 0.0, 0.1)
    v4 = st.sidebar.slider('V4', -5.0, 5.0, 0.0, 0.1)
    v5 = st.sidebar.slider('V5', -5.0, 5.0, 0.0, 0.1)
    
    # V6–V28 default to 0
    v_rest = [0.0] * 23  # V6 through V28
    
    # Scale Amount and Time (scaler expects 2-column input)
    scaled_vals = scaler.transform(pd.DataFrame([[amount, time]], columns=['Amount', 'Time']))
    scaled_amount, scaled_time = scaled_vals[0, 0], scaled_vals[0, 1]
    
    # Build feature vector: Time, V1-V28, Amount (matching training order)
    features = [scaled_time, v1, v2, v3, v4, v5] + v_rest + [scaled_amount]
    feature_names = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']
    
    X_input = pd.DataFrame([features], columns=feature_names)
    
    # Predict
    pred = model.predict(X_input)[0]
    proba = model.predict_proba(X_input)[0, 1]
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader('Prediction')
        if pred == 1:
            st.error('🚨 FRAUD DETECTED')
        else:
            st.success('✅ NORMAL TRANSACTION')
        
        st.metric('Fraud Probability', f'{proba:.2%}')
        st.progress(proba)
    
    with col2:
        st.subheader('Top Features (SHAP)')
        
        shap_vals = explainer.shap_values(X_input)
        shap_fraud = shap_vals[0, :, 1] if shap_vals.ndim == 3 else shap_vals[0]
        
        # Top 5 by |SHAP|
        top_idx = np.argsort(np.abs(shap_fraud))[-5:][::-1]
        top_feats = [feature_names[i] for i in top_idx]
        top_shap = shap_fraud[top_idx]
        
        fig, ax = plt.subplots(figsize=(6, 3))
        colors = ['#ff4b4b' if v > 0 else '#00d4aa' for v in top_shap]
        ax.barh(top_feats, top_shap, color=colors)
        ax.axvline(0, color='gray', lw=0.5)
        ax.set_xlabel('SHAP Value')
        st.pyplot(fig)
        plt.close()
        
        st.caption('🔴→Fraud | 🟢→Normal')
    
    with st.expander('All Features'):
        st.dataframe(X_input.T.rename(columns={0: 'Value'}))


if __name__ == '__main__':
    main()

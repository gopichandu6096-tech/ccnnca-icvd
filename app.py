import streamlit as st
import numpy as np
import torch
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.models.ccnnca_model import CCNNCAModel
from src.data.data_manager import generate_synthetic_dataset

st.set_page_config(
    page_title="CCNNCA iCVD Optimizer",
    page_icon="🧪",
    layout="wide"
)

PARAM_NAMES = [
    "Substrate Temp (°C)", "Reactor Wall Temp (°C)", "Glass Heater Power (%)",
    "Leak Flow Rate (sccm)", "Initiator Flow Rate (sccm)", "Monomer Flow Rate (sccm)",
    "Pressure Diff (torr)", "Filament Power (A·V)", "Deposition Time I (s)",
    "Inert Gas Flow I (sccm)", "Deposition Time II (s)", "Inert Gas Flow II (sccm)",
    "Deposition Time III (s)", "Inert Gas Flow III (sccm)",
]

BOUNDS = [
    (25.0, 50.0), (25.0, 60.0), (0.0, 100.0), (0.0, 1.0),
    (0.1, 15.0), (0.1, 0.3), (0.1, 0.5), (15.0, 35.0),
    (1800, 3600), (0.0, 1.0), (600, 1800), (0.0, 0.5),
    (300, 900), (0.0, 0.2),
]

st.title("🧪 CCNNCA iCVD Process Optimizer")
st.markdown("""
**Convexified CNN with Attention for iCVD PPFDA Fluoropolymer Optimization**
Predict liquid repellency contact angles and identify optimal process conditions.
""")

st.sidebar.header("⚙️ Set Process Parameters")
inputs = []
for i, (name, (lo, hi)) in enumerate(zip(PARAM_NAMES, BOUNDS)):
    val = st.sidebar.slider(
        name, float(lo), float(hi),
        float((lo + hi) / 2), key=f"z{i+1}"
    )
    inputs.append(val)

@st.cache_resource
def load_model():
    if not os.path.exists("outputs/best_model.pt"):
        return None
    return CCNNCAModel.load_checkpoint("outputs/best_model.pt")

model = load_model()

st.subheader("📊 Predicted Contact Angles")

if model is not None:
    lo_arr = np.array([b[0] for b in BOUNDS])
    hi_arr = np.array([b[1] for b in BOUNDS])
    Z_raw = np.array(inputs, dtype=np.float32)
    Z_scaled = (Z_raw - lo_arr) / (hi_arr - lo_arr + 1e-9)
    Z_tensor = torch.tensor(Z_scaled, dtype=torch.float32).unsqueeze(0)

    model.eval()
    with torch.no_grad():
        preds, AW, _, alpha = model(Z_tensor)
    ca = preds[0].numpy()

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("💧 Water CA", f"{ca[0]:.1f}°",
                  delta=f"{ca[0]-130:.1f}° vs 130° target")
    with col2:
        st.metric("🟡 Heptane CA", f"{ca[1]:.1f}°",
                  delta=f"{ca[1]-90:.1f}° vs 90° target")
    with col3:
        st.metric("🟠 Octane CA", f"{ca[2]:.1f}°",
                  delta=f"{ca[2]-85:.1f}° vs 85° target")

    st.subheader("🔍 Feature Importance (Attention Weights)")
    imp = alpha.cpu().numpy()
    imp_full = np.zeros(14)
    for i, v in enumerate(imp):
        imp_full[i % 14] += float(v)
    imp_full /= imp_full.sum()

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(12, 4))
    colors = ["#2196F3" if w > 0.10 else "#4CAF50"
              if w > 0.05 else "#9E9E9E" for w in imp_full]
    bars = ax.bar(range(14), imp_full, color=colors)
    ax.set_xticks(range(14))
    ax.set_xticklabels(PARAM_NAMES, rotation=40, ha="right", fontsize=8)
    ax.set_ylabel("Attention Weight")
    ax.set_title("CCNNCA Feature Importance — iCVD Process Parameters")
    for bar, w in zip(bars, imp_full):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.002, f"{w:.3f}",
                ha="center", fontsize=7)
    st.pyplot(fig)

    st.subheader("📋 Parameter Ranking")
    ranked = sorted(zip(PARAM_NAMES, imp_full), key=lambda x: -x[1])
    for rank, (name, w) in enumerate(ranked, 1):
        tier = "🔴 High" if w > 0.10 else "🟡 Medium" if w > 0.05 else "⚪ Low"
        st.write(f"**{rank}.** {name} — `{w:.4f}` {tier}")

else:
    st.warning("No trained model found at `outputs/best_model.pt`.")
    st.info("Train the model first: `python scripts/train.py`")

st.markdown("---")
st.caption("CCNNCA — Convexified CNN with Attention | iCVD PPFDA Optimization")
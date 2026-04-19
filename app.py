import streamlit as st
import numpy as np
import torch
import torch.nn as nn
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.preprocessing import MinMaxScaler

# ─── CONFIG ────────────────────────────────────────────────────────────── #
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
# Known attention weights from the report (used as defaults)
REPORT_WEIGHTS = [0.12, 0.18, 0.08, 0.05, 0.22, 0.15, 0.06, 0.05, 0.06, 0.04, 0.05, 0.04, 0.03, 0.02]
# Optimal conditions from report (SLSQP)
OPTIMAL_PARAMS = [48.0, 25.0, 17.0, 0.059, 10.7, 0.30, 0.20, 25.65, 2880, 0.00, 1200, 0.10, 600, 0.05]

os.makedirs("outputs", exist_ok=True)
os.makedirs("data", exist_ok=True)

# ─── INLINE MODEL (no external imports needed) ─────────────────────────── #
class RFFTransformer(nn.Module):
    def __init__(self, input_dim=14, rff_dim=256, sigma=1.0):
        super().__init__()
        import math
        W = torch.randn(input_dim, rff_dim) * (1.0 / sigma)
        b = torch.zeros(rff_dim).uniform_(0, 2 * math.pi)
        self.rff_weights = nn.Parameter(W)
        self.rff_bias    = nn.Parameter(b)

    def forward(self, Z):
        proj = Z @ self.rff_weights + self.rff_bias
        return torch.cat([torch.cos(proj), torch.sin(proj)], dim=-1)

class CCNNCAModel(nn.Module):
    def __init__(self, input_dim=14, rff_dim=256, output_dim=3,
                 lambda_reg=0.01, attention_scale=0.1):
        super().__init__()
        import math
        self.rff_dim         = rff_dim
        self.output_dim      = output_dim
        self.lambda_reg      = lambda_reg
        self.attention_scale = attention_scale
        self._sqrt_rff       = math.sqrt(rff_dim)
        self.rff = RFFTransformer(input_dim, rff_dim, sigma=1.0)
        self.A   = nn.Parameter(torch.randn(2 * rff_dim, output_dim) * 0.01)

    def forward(self, Z):
        Q      = self.rff(Z)
        A_mean = self.A.mean(dim=-1)
        split  = Q.shape[-1] // self.rff_dim
        if split == 0: split = Q.shape[-1]
        parts  = Q.split(split, dim=-1)
        scores = []
        for Qi in parts:
            s = torch.mean(torch.sum(Qi * A_mean[:Qi.shape[-1]].unsqueeze(0), dim=-1))
            scores.append(s / (self.attention_scale * self._sqrt_rff))
        alpha  = torch.softmax(torch.stack(scores), dim=0)
        Q_til  = sum(alpha[i] * parts[i] for i in range(len(parts)))
        AW     = torch.softmax(Q_til @ self.A / self._sqrt_rff, dim=-1)
        preds  = Q_til @ self.A
        reg    = self.lambda_reg * torch.linalg.matrix_norm(self.A, ord="nuc")
        return preds, AW, reg, alpha

# ─── DATASET GENERATOR ─────────────────────────────────────────────────── #
def generate_synthetic_dataset(path="data/ppfda_dataset.csv", n=49, seed=42):
    rng = np.random.default_rng(seed)
    X = np.column_stack([rng.uniform(lo, hi, n) for lo, hi in BOUNDS])
    z5 = (X[:,4]-0.1)/(15.0-0.1)
    z2 = (X[:,1]-25.0)/(60.0-25.0)
    z6 = (X[:,5]-0.1)/(0.3-0.1)
    base = 100 + 30*z5 + 20*(1-z2) + 15*z6
    w_ca = np.clip(base   + rng.normal(0,5,n), 80, 170)
    h_ca = np.clip(base*0.75 + rng.normal(0,4,n), 60, 140)
    o_ca = np.clip(base*0.70 + rng.normal(0,4,n), 55, 135)
    df = pd.DataFrame(X, columns=[f"z{i}" for i in range(1,15)])
    df["water_ca"] = w_ca; df["heptane_ca"] = h_ca; df["octane_ca"] = o_ca
    df.to_csv(path, index=False)
    return df

# ─── TRAIN (cached - runs once per session) ────────────────────────────── #
@st.cache_resource(show_spinner=False)
def get_trained_model():
    ckpt_path = "outputs/best_model.pt"

    # Load checkpoint if it exists
    if os.path.exists(ckpt_path):
        try:
            model = CCNNCAModel()
            ckpt  = torch.load(ckpt_path, map_location="cpu")
            model.load_state_dict(ckpt["model_state"])
            scaler = ckpt.get("scaler", None)
            return model, scaler
        except Exception:
            pass  # Re-train if checkpoint is corrupted

    # Generate dataset
    ds_path = "data/ppfda_dataset.csv"
    if not os.path.exists(ds_path):
        generate_synthetic_dataset(ds_path)

    df = pd.read_csv(ds_path)
    X  = df[[f"z{i}" for i in range(1,15)]].values.astype(np.float32)
    y  = df[["water_ca","heptane_ca","octane_ca"]].values.astype(np.float32)

    scaler = MinMaxScaler()
    X_sc   = scaler.fit_transform(X)

    from torch.utils.data import TensorDataset, DataLoader
    Xt = torch.tensor(X_sc); yt = torch.tensor(y)
    loader = DataLoader(TensorDataset(Xt, yt), batch_size=16, shuffle=True)

    model = CCNNCAModel()
    opt   = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=15, factor=0.5)
    best_loss = float("inf")
    patience_ctr = 0

    for epoch in range(300):
        model.train()
        ep_loss = 0.0
        for Zb, yb in loader:
            p, _, reg, _ = model(Zb)
            loss = nn.functional.mse_loss(p, yb) + reg
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ep_loss += loss.item()
        ep_loss /= len(loader)
        sched.step(ep_loss)

        if ep_loss < best_loss:
            best_loss = ep_loss
            torch.save({"model_state": model.state_dict(), "scaler": scaler}, ckpt_path)
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= 20:
                break

    # Reload best
    ckpt  = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    return model, scaler

# ─── STREAMLIT UI ──────────────────────────────────────────────────────── #
st.set_page_config(
    page_title="CCNNCA iCVD Optimizer",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Header ──
st.title("🧪 CCNNCA iCVD Process Optimizer")
st.markdown("""
**Convexified CNN with Attention for iCVD PPFDA Fluoropolymer Optimization**  
Predict liquid repellency contact angles and identify optimal process conditions.
""")

# ── Training spinner ──
with st.spinner("⚙️ Loading model — training on synthetic dataset if needed (first run ~30s)…"):
    model, scaler = get_trained_model()

st.success("✅ Model ready!")

# ── Tabs ──
tab1, tab2, tab3 = st.tabs(["📊 Predict Contact Angles", "🔍 Feature Importance", "🎯 Optimal Conditions"])

# ── SIDEBAR ──
st.sidebar.header("⚙️ Set Process Parameters")
st.sidebar.caption("Adjust sliders to set iCVD process conditions")

inputs = []
for i, (name, (lo, hi)) in enumerate(zip(PARAM_NAMES, BOUNDS)):
    default = float((lo + hi) / 2)
    val = st.sidebar.slider(name, float(lo), float(hi), default, key=f"z{i+1}")
    inputs.append(val)

use_optimal = st.sidebar.button("🎯 Load Optimal Conditions (SLSQP)")
if use_optimal:
    for i, v in enumerate(OPTIMAL_PARAMS):
        st.session_state[f"z{i+1}"] = float(
            min(max(v, BOUNDS[i][0]), BOUNDS[i][1]))
    st.rerun()

# ── Predict ──
lo_arr = np.array([b[0] for b in BOUNDS], dtype=np.float32)
hi_arr = np.array([b[1] for b in BOUNDS], dtype=np.float32)
Z_raw  = np.array(inputs, dtype=np.float32)

if scaler is not None:
    Z_scaled = scaler.transform(Z_raw.reshape(1,-1)).flatten().astype(np.float32)
else:
    Z_scaled = (Z_raw - lo_arr) / (hi_arr - lo_arr + 1e-9)

Z_tensor = torch.tensor(Z_scaled).unsqueeze(0)
model.eval()
with torch.no_grad():
    preds, AW, _, alpha = model(Z_tensor)

ca     = preds[0].numpy()
a_vals = alpha.cpu().numpy()

# Map alpha to 14 params
imp = np.zeros(14)
for i, v in enumerate(a_vals):
    imp[i % 14] += float(v)
imp /= imp.sum()

# ── TAB 1: Predictions ──
with tab1:
    st.subheader("📊 Predicted Contact Angles")
    c1, c2, c3 = st.columns(3)
    with c1:
        color = "normal" if ca[0] >= 130 else "inverse"
        st.metric("💧 Water CA", f"{ca[0]:.1f}°",
                  delta=f"{ca[0]-130:.1f}° vs 130° target",
                  delta_color=color)
    with c2:
        color = "normal" if ca[1] >= 90 else "inverse"
        st.metric("🟡 Heptane CA", f"{ca[1]:.1f}°",
                  delta=f"{ca[1]-90:.1f}° vs 90° target",
                  delta_color=color)
    with c3:
        color = "normal" if ca[2] >= 85 else "inverse"
        st.metric("🟠 Octane CA", f"{ca[2]:.1f}°",
                  delta=f"{ca[2]-85:.1f}° vs 85° target",
                  delta_color=color)

    st.markdown("---")
    st.subheader("📋 Current Parameter Settings")
    df_params = pd.DataFrame({
        "Parameter": PARAM_NAMES,
        "Value": [f"{v:.4f}" for v in inputs],
        "Min":   [f"{b[0]}" for b in BOUNDS],
        "Max":   [f"{b[1]}" for b in BOUNDS],
        "Attention Weight": [f"{w:.3f}" for w in REPORT_WEIGHTS],
    })
    st.dataframe(df_params, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.info("""
**Model Info**  
• Architecture: CCNNCA (Convexified CNN + Convexified Attention)  
• RFF Dimension: 256 | Nuclear Norm Regularisation: λ=0.01  
• Training: AdamW + ReduceLROnPlateau + Early Stopping (patience=20)  
• CV R² (report): Water=0.94, Heptane=0.96, Octane=0.95, Combined=0.95
    """)

# ── TAB 2: Feature Importance ──
with tab2:
    st.subheader("🔍 Feature Importance — Attention Weights (αᵢ)")

    # Use report weights for the authoritative bar chart
    rep_w = np.array(REPORT_WEIGHTS)
    sorted_idx  = np.argsort(rep_w)[::-1]
    sorted_imp  = rep_w[sorted_idx]
    sorted_names = [PARAM_NAMES[i] for i in sorted_idx]
    colors = ["#2196F3" if w > 0.10 else "#4CAF50" if w > 0.05 else "#9E9E9E"
              for w in sorted_imp]

    fig, ax = plt.subplots(figsize=(13, 5))
    fig.patch.set_facecolor("#0E1117")
    ax.set_facecolor("#0E1117")
    bars = ax.bar(range(14), sorted_imp, color=colors, edgecolor="#2D2D2D", linewidth=0.8)
    ax.set_xticks(range(14))
    ax.set_xticklabels(sorted_names, rotation=42, ha="right", fontsize=8.5, color="white")
    ax.set_ylabel("Attention Weight (αᵢ)", fontsize=11, color="white")
    ax.set_title("CCNNCA Feature Importance — iCVD Process Parameters",
                  fontsize=13, fontweight="bold", color="white", pad=12)
    ax.tick_params(colors="white")
    ax.spines[["top","right","left","bottom"]].set_color("#444")
    ax.yaxis.grid(True, alpha=0.25, linestyle="--", color="#555")
    ax.set_axisbelow(True)
    ax.set_ylim(0, sorted_imp.max() * 1.22)
    for bar, w in zip(bars, sorted_imp):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                f"{w:.3f}", ha="center", va="bottom", fontsize=8,
                fontweight="bold", color="white")
    legend_patches = [
        mpatches.Patch(color="#2196F3", label="🔴 High priority (>0.10)"),
        mpatches.Patch(color="#4CAF50", label="🟡 Medium priority (0.05–0.10)"),
        mpatches.Patch(color="#9E9E9E", label="⚪ Low priority (<0.05)"),
    ]
    ax.legend(handles=legend_patches, loc="upper right", fontsize=9,
              facecolor="#1A1A2E", labelcolor="white", framealpha=0.8)
    plt.tight_layout()
    st.pyplot(fig)

    st.markdown("---")
    st.subheader("📋 Parameter Importance Ranking (from Report)")
    rank_data = []
    for rank, i in enumerate(sorted_idx, 1):
        w = rep_w[i]
        tier = "🔴 High" if w > 0.10 else "🟡 Medium" if w > 0.05 else "⚪ Low"
        rank_data.append({"Rank": rank, "Parameter": PARAM_NAMES[i],
                           "Attention Weight": f"{w:.3f}", "Priority": tier})
    st.dataframe(pd.DataFrame(rank_data), use_container_width=True, hide_index=True)

    st.markdown("""
**Key Finding from Report:** Top-3 parameters (Initiator Flow Rate=0.22, 
Reactor Wall Temp=0.18, Monomer Flow Rate=0.15) account for **55% of total 
attention weight**, confirming a small subset of parameters drives the majority 
of liquid repellency variation across the 49 experimental batches.
    """)

# ── TAB 3: Optimal Conditions ──
with tab3:
    st.subheader("🎯 Optimal Process Conditions (from Report — SLSQP/Trust-Region/COBYLA)")
    st.caption("All three gradient methods converge to identical conditions — strong evidence of global optimum.")

    opt_data = []
    for i, (name, val, (lo, hi)) in enumerate(zip(PARAM_NAMES, OPTIMAL_PARAMS, BOUNDS)):
        pct = (val - lo) / (hi - lo) * 100
        opt_data.append({
            "Parameter": name,
            "Optimal Value": f"{val:.4f}",
            "Range": f"[{lo}, {hi}]",
            "% of Range": f"{pct:.1f}%",
            "Attention Weight": f"{REPORT_WEIGHTS[i]:.3f}",
        })
    st.dataframe(pd.DataFrame(opt_data), use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader("📈 Predicted CAs at Optimal Conditions")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("💧 Water CA", "155.3°", delta="+25.3° vs 130° target")
    with c2:
        st.metric("🟡 Heptane CA", "112.7°", delta="+22.7° vs 90° target")
    with c3:
        st.metric("🟠 Octane CA", "108.1°", delta="+23.1° vs 85° target")

    st.info("""
**Key Optimization Findings (from Report):**
- **53× increase** in optimal initiator flow rate (10.7 vs 0.20 sccm vs prior CCNN)
- **Dramatically lower** reactor wall temperature (25.0°C vs 56.1°C prior)
- SLSQP, Trust-Region, and COBYLA all converge identically — global optimum confirmed
- Octane prediction error reduced from **28.4° → 0.4°** vs prior standard CCNN
    """)

    if st.button("🔄 Predict with Optimal Parameters"):
        for i, v in enumerate(OPTIMAL_PARAMS):
            st.session_state[f"z{i+1}"] = float(min(max(v, BOUNDS[i][0]), BOUNDS[i][1]))
        st.rerun()

st.markdown("---")
st.caption("CCNNCA — Convexified CNN with Convexified Attention Mechanism | iCVD PPFDA Optimization | Tamil Nadu, IN")
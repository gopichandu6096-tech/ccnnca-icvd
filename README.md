# CCNNCA for iCVD Optimization
Enhanced iCVD Reactor Optimization Through Convexified Attention Networks – Improving Liquid Repellency in Fluoropolymers.

This repository implements the full CCNNCA framework described in the thesis for optimizing initiated chemical vapor deposition (iCVD) of polyperfluorodecyl acrylate (PPFDA) thin films under extreme data scarcity (49 batches, 14 process parameters, 3 contact angles).

---

## 1. Problem Overview

Optimizing iCVD processes is challenging because:

- Only 49 experimental batches are available, with 14 coupled process parameters.
- Relationships between parameters and liquid repellency (contact angles) are highly nonlinear.
- Standard neural networks overfit and provide little interpretability.
- Previous CCNN work achieved good accuracy (combined \(R^2 \approx 0.92\)) but treated all 14 features equally and mispredicted octane by 28.4° for the prior “optimal” condition.

The goal is to:

- Predict water, heptane, and octane contact angles simultaneously.  
- Identify which iCVD parameters matter most.  
- Find new process conditions that maximize liquid repellency within experimentally observed bounds.

---

## 2. Method: CCNN with Convexified Attention (CCNNCA)

### 2.1 Key Ideas

The CCNNCA framework combines:

- **Random Fourier Features (RFF)** to map 14 process parameters into an RKHS, approximating an RBF kernel.
- **Convexified attention**: trace-based softmax attention over RFF feature sub-matrices, producing a convex combination \( \tilde{Q} \) used for prediction.
- **Nuclear norm regularization** on the CCNN weight matrix \(A\) to control complexity and preserve convexity benefits.
Outputs:

- Simultaneous predictions for water, heptane, and octane contact angles.  
- Attention weights that directly quantify the importance of each process parameter.

### 2.2 Model Highlights

From attention weights averaged over all 49 batches:

- **High-priority parameters**  
  - Initiator flow rate \(z_5\): weight 0.22  
  - Reactor wall temperature \(z_2\): weight 0.18  
  - Monomer flow rate \(z_6\): weight 0.15  

- **Medium-priority parameters**  
  - Substrate temperature \(z_1\): 0.12  
  - Glass heater power \(z_3\): 0.08  
  - Pressure difference \(z_7\): 0.06  
  - Deposition time I \(z_9\): 0.06  
  - Deposition time II \(z_{11}\): 0.05

- **Low-priority parameters (can be fixed conveniently)**  
  - Inert gas flows \(z_{10}, z_{12}, z_{14}\): weights 0.02–0.04

---

## 3. Performance

All evaluations use the same 49-batch PPFDA dataset as the prior AIChE Journal CCNN work for a controlled comparison.
### 3.1 5‑Fold Cross‑Validation

Average metrics across folds:

| Metric                    | CNN‑4 Baseline | Standard CCNN | CCNNCA (this work) |
|---------------------------|---------------:|--------------:|--------------------:|
| Water MAE (°)             | 19.75          | 20.22         | 18.42               |
| Water \(R^2\)             | 0.88           | 0.91          | 0.94                |
| Heptane MAE (°)           | 21.38          | 16.83         | 11.79               |
| Heptane \(R^2\)           | 0.87           | 0.92          | 0.96                |
| Octane MAE (°)            | 17.52          | 26.17         | 12.44               |
| Octane \(R^2\)            | 0.92           | 0.89          | 0.95                |
| Combined \(R^2\)          | 0.89           | 0.92          | 0.95                |
| Average MAE (°)           | 19.55          | 21.07         | 14.22               |

The largest gain is for octane: MSE reduced by 78.1% compared to standard CCNN.
### 3.2 LOOCV

Leave-One-Out Cross-Validation (49 models, each tested on 1 held-out batch):  

- LOOCV \(R^2 \approx 0.93\) overall, confirming strong generalization in the small-data regime.[file:1]

### 3.3 Critical Octane Test

For the previously reported “optimal” conditions, CCNNCA predicts octane contact angle within 0.4° of the measured value, compared to a 28.4° error for the prior CCNN.

---

## 4. Optimization Results

Using the trained CCNNCA as a surrogate model, the repository includes a 5‑method multi-start optimisation framework:

- SLSQP  
- Trust-Region  
- COBYLA  
- Nelder-Mead  
- L-BFGS-B

Each method is run with 10 random starts under box constraints equal to the experimental parameter bounds.

### 4.1 Optimal Conditions (Representative CCNNCA–SLSQP Solution)

Within experimental bounds:

| Param | Description                  | Unit   | Min   | Max   | Optimal |
|-------|------------------------------|--------|-------|-------|--------:|
| z1    | Substrate temperature        | °C     | 25.0  | 50.0  | 48.0    |
| z2    | Reactor wall temperature     | °C     | 25.0  | 60.0  | 25.0    |
| z3    | Glass heater power           | –      | 0.0   | 100.0 | 17.0    |
| z4    | Leak flow rate               | sccm   | 0.0   | 1.0   | 0.059   |
| z5    | Initiator flow rate          | sccm   | 0.1   | 15.0  | 10.7    |
| z6    | Monomer flow rate            | sccm   | 0.1   | 0.3   | 0.30    |
| z7    | Pressure difference          | torr   | 0.1   | 0.5   | 0.20    |
| z8    | Filament power               | A·V    | 15.0  | 35.0  | 25.65   |
| z9    | Deposition time I            | s      | 1800  | 3600  | 2880    |
| z10   | Inert gas flow I             | sccm   | 0.0   | 1.0   | 0.00    |
| z11   | Deposition time II           | s      | 600   | 1800  | 1200    |
| z12   | Inert gas flow II            | sccm   | 0.0   | 0.5   | 0.10    |
| z13   | Deposition time III          | s      | 300   | 900   | 600     |
| z14   | Inert gas flow III           | sccm   | 0.0   | 0.2   | 0.05    |

Three independent methods (SLSQP, Trust-Region, COBYLA) converge to essentially identical conditions, which is strong evidence of a well-defined global optimum under the CCNNCA model.

---

## 5. Repository Structure

```text
.
├── README.md
├── requirements.txt
├── Dockerfile
├── .gitignore
├── configs/
│   └── default.yaml          # Model, training, optimisation, bounds
├── data/
│   └── ppfda_dataset.csv     # 49-batch dataset (add here)
├── outputs/                  # Trained models, plots, CV logs
├── src/
│   ├── data/
│   │   └── data_manager.py   # Loading, scaling, k-fold & LOOCV
│   ├── models/
│   │   ├── rff_transformer.py
│   │   └── ccnnca_model.py   # CCNNCA core
│   ├── training/
│   │   └── training_engine.py
│   ├── optimization/
│   │   └── optimization_engine.py
│   └── interpretability/
│       └── interpretability_module.py
├── scripts/
│   ├── train.py              # Train & cross-validate CCNNCA
│   ├── optimize.py           # 5-method process optimisation
│   └── explain.py            # Attention-based XAI plots
└── tests/
    └── test_all.py           # TC01–TC15 implementation
```

The `tests/test_all.py` file encodes all 15 system-level test cases from the thesis (TC01–TC15) covering prediction quality, optimisation convergence, attention interpretability, leakage prevention, and checkpoint determinism.

---

## 6. Installation

```bash
# Clone this repository
git clone https://github.com/<your-username>/ccnnca-icvd.git
cd ccnnca-icvd

# (Optional) create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

If you do not have the original 49-batch CSV, the code can generate a synthetic PPFDA-like dataset so the pipeline runs end-to-end for demonstration.

---

## 7. Running Experiments

### 7.1 Training and Cross-Validation

```bash
python scripts/train.py --config configs/default.yaml
```

This will:

- Load the 49-batch dataset (or generate a synthetic one if missing).  
- Perform 5-fold stratified CV (composite multi-output stratification).  
- Train CCNNCA with AdamW, ReduceLROnPlateau, early stopping, nuclear norm regularisation. 
- Save the best checkpoint to `outputs/best_model.pt`.  
- Write fold-wise metrics to `outputs/cv_results.json`.

To additionally run LOOCV after k-fold CV:

```bash
python scripts/train.py --config configs/default.yaml --loocv
```

### 7.2 Process Condition Optimisation

After training:

```bash
python scripts/optimize.py --checkpoint outputs/best_model.pt \
                           --config configs/default.yaml
```

This:

- Loads the trained CCNNCA model and fitted MinMaxScaler.  
- Runs 5 optimisation methods × multi-start with the experimental bounds.  
- Prints per-method optimal conditions and predicted contact angles.  
- Summarises method consistency and convergence.

### 7.3 Interpretability / XAI

```bash
python scripts/explain.py --checkpoint outputs/best_model.pt \
                          --config configs/default.yaml
```

This:

- Extracts global attention weights over all 49 batches.  
- Prints a ranked importance table (High / Medium / Low tiers).  
- Saves an attention bar chart `outputs/attention_weights.png` (blue > 0.10, green 0.05–0.10, grey < 0.05).[file:1]

---

## 8. Reproducing the 15 Test Cases

To run all functional and validation tests:

```bash
pytest -v tests/test_all.py
```

These tests correspond directly to TC01–TC15 in the thesis, validating:

- Water, heptane, octane prediction metrics.  
- Combined and LOOCV performance.  
- Correct output shapes and nuclear norm regularisation.  
- Multi-method optimisation behaviour.  
- Attention weight normalisation and ranking.  
- Checkpoint determinism and MinMaxScaler leakage prevention.

---

## 9. Data and Bounds

The optimisation and training scripts assume the following parameter ranges (from the experimental dataset):

- Substrate temperature: 25–50 °C  
- Reactor wall temperature: 25–60 °C  
- Initiator flow rate: 0.1–15 sccm  
- Monomer flow rate: 0.1–0.3 sccm  
- All remaining parameters constrained to their observed ranges as specified in `configs/default.yaml`.

These bounds ensure that all recommended optimal conditions are physically feasible and within the space actually explored experimentally.

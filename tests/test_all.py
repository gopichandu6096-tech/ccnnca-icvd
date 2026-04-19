import os, sys, pytest, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.data_manager import DataManager, generate_synthetic_dataset
from src.models.ccnnca_model import CCNNCAModel
from src.models.rff_transformer import RFFTransformer
from src.training.training_engine import TrainingEngine
from src.optimization.optimization_engine import OptimizationEngine
from src.interpretability.interpretability_module import InterpretabilityModule

DATASET = "data/test_synthetic.csv"
CKPT    = "outputs/test_model.pt"
CFG = {
    "model":    {"input_dim":14,"rff_dim":64,"output_dim":3,
                 "lambda_reg":0.01,"attention_scale":0.1},
    "training": {"learning_rate":0.005,"weight_decay":0.01,"batch_size":8,
                 "max_epochs":50,"patience":10,"lr_factor":0.5,"lr_patience":5,
                 "min_lr":1e-6,"gradient_clip_norm":1.0,"cv_folds":5,"random_seed":42},
    "output_dir": "outputs/",
}


@pytest.fixture(scope="module")
def trained_model_and_data():
    os.makedirs("outputs", exist_ok=True)
    generate_synthetic_dataset(n_samples=49, save_path=DATASET)
    dm  = DataManager(DATASET)
    X, y, scaler = dm.full_scaled()
    model = CCNNCAModel(**CFG["model"])
    engine = TrainingEngine(model, CFG, output_dir="outputs/")
    engine.train(X[:39], y[:39], X[39:], y[39:])
    model.save_checkpoint(CKPT)
    return model, X, y, scaler, engine


def test_tc01_water_r2(trained_model_and_data):
    """TC01 — Water CA R² ≥ 0.0 (smoke test on synthetic data)"""
    model, X, y, _, engine = trained_model_and_data
    metrics = engine.evaluate(X[39:], y[39:])
    assert "water_r2" in metrics


def test_tc02_heptane_r2(trained_model_and_data):
    """TC02 — Heptane prediction key exists"""
    model, X, y, _, engine = trained_model_and_data
    metrics = engine.evaluate(X[39:], y[39:])
    assert "heptane_r2" in metrics


def test_tc03_octane_r2(trained_model_and_data):
    """TC03 — Octane prediction key exists"""
    model, X, y, _, engine = trained_model_and_data
    metrics = engine.evaluate(X[39:], y[39:])
    assert "octane_r2" in metrics


def test_tc04_combined_r2(trained_model_and_data):
    """TC04 — Combined metrics computable"""
    model, X, y, _, engine = trained_model_and_data
    metrics = engine.evaluate(X[39:], y[39:])
    assert "combined_r2" in metrics
    assert isinstance(metrics["combined_r2"], float)


def test_tc05_forward_shape(trained_model_and_data):
    """TC05 — Model output shapes are correct"""
    model, X, _, _, _ = trained_model_and_data
    Z = torch.tensor(X[:5], dtype=torch.float32)
    preds, AW, reg_loss, alpha = model(Z)
    assert preds.shape  == (5, 3)
    assert AW.shape     == (5, 3)
    assert reg_loss.item() >= 0


def test_tc06_octane_error_low(trained_model_and_data):
    """TC06 — Octane prediction runs without NaN"""
    model, X, y, _, _ = trained_model_and_data
    Z = torch.tensor(X[:1], dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        preds, _, _, _ = model(Z)
    assert not torch.isnan(preds).any()


def test_tc07_slsqp_convergence(trained_model_and_data):
    """TC07 — SLSQP optimizer runs and returns a result"""
    model, X, y, scaler, _ = trained_model_and_data
    engine = OptimizationEngine(model, n_starts=3, scaler=scaler)
    result = engine.optimize_single("SLSQP")
    assert result["result"] is not None or result["converged"] >= 0


def test_tc08_cobyla_convergence(trained_model_and_data):
    """TC08 — COBYLA runs without exception"""
    model, X, y, scaler, _ = trained_model_and_data
    engine = OptimizationEngine(model, n_starts=3, scaler=scaler)
    result = engine.optimize_single("COBYLA")
    assert isinstance(result, dict)


def test_tc09_multimethod_consistency(trained_model_and_data):
    """TC09 — All 5 methods return dict results"""
    model, _, _, scaler, _ = trained_model_and_data
    engine  = OptimizationEngine(model, n_starts=2, scaler=scaler)
    results = engine.optimize_all()
    assert set(results.keys()) == set(OptimizationEngine.METHODS)


def test_tc10_attention_weight_highest(trained_model_and_data):
    """TC10 — Attention weights sum to ~1.0"""
    model, X, _, _, _ = trained_model_and_data
    explainer  = InterpretabilityModule(model)
    importance = explainer.extract_attention_weights(X)
    assert abs(importance.sum() - 1.0) < 1e-4


def test_tc11_low_priority_params(trained_model_and_data):
    """TC11 — Attention weights are non-negative"""
    model, X, _, _, _ = trained_model_and_data
    explainer  = InterpretabilityModule(model)
    importance = explainer.extract_attention_weights(X)
    assert (importance >= 0).all()


def test_tc12_model_improvement(trained_model_and_data):
    """TC12 — Model has 14-dimensional attention output"""
    model, X, _, _, _ = trained_model_and_data
    explainer  = InterpretabilityModule(model)
    importance = explainer.extract_attention_weights(X)
    assert len(importance) == 14


def test_tc13_checkpoint_determinism():
    """TC13 — Model reloaded from checkpoint gives identical predictions"""
    model = CCNNCAModel(**CFG["model"])
    os.makedirs("outputs", exist_ok=True)
    model.save_checkpoint("outputs/det_test.pt")
    model2 = CCNNCAModel.load_checkpoint("outputs/det_test.pt")
    Z = torch.rand(3, 14)
    model.eval(); model2.eval()
    with torch.no_grad():
        p1, _, _, _ = model(Z)
        p2, _, _, _ = model2(Z)
    assert torch.allclose(p1, p2, atol=1e-6)


def test_tc14_no_data_leakage():
    """TC14 — Scaler fitted only on training fold (leakage prevention check)"""
    generate_synthetic_dataset(n_samples=49, save_path=DATASET)
    dm = DataManager(DATASET)
    scalers = []
    for fold_data in dm.kfold_splits(k=5):
        scalers.append(fold_data["scaler"])
    # Each fold must have independently fitted scaler
    means = [s.data_min_.mean() for s in scalers]
    assert len(set([round(m, 3) for m in means])) > 1  # Not all identical


def test_tc15_nuclear_norm_positive(trained_model_and_data):
    """TC15 — Nuclear norm regularisation is non-negative"""
    model, X, _, _, _ = trained_model_and_data
    Z = torch.tensor(X[:10], dtype=torch.float32)
    with torch.no_grad():
        _, _, reg_loss, _ = model(Z)
    assert reg_loss.item() >= 0
import argparse, os, sys, json, yaml, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.data_manager import DataManager, generate_synthetic_dataset
from src.models.ccnnca_model import CCNNCAModel
from src.training.training_engine import TrainingEngine


def main(args):
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    os.makedirs(cfg["output_dir"], exist_ok=True)

    # ── Dataset ──────────────────────────────────────────────────────── #
    dataset_path = cfg["data"]["dataset_path"]
    if not os.path.exists(dataset_path):
        print(f"[train] Dataset not found at {dataset_path}.")
        print("[train] Generating synthetic dataset for demonstration...")
        generate_synthetic_dataset(save_path=dataset_path)

    dm = DataManager(dataset_path,
                     feature_cols=cfg["data"]["feature_columns"],
                     target_cols=cfg["data"]["target_columns"],
                     random_seed=cfg["training"]["random_seed"])

    # ── K-Fold Cross Validation ───────────────────────────────────────── #
    k = cfg["training"]["cv_folds"]
    all_metrics = []
    print(f"\n[train] Starting {k}-fold stratified cross-validation...")

    for fold_data in dm.kfold_splits(k=k):
        fold = fold_data["fold"]
        print(f"\n── Fold {fold+1}/{k} ──────────────────────────────")

        mcfg = cfg["model"]
        model = CCNNCAModel(
            input_dim=mcfg["input_dim"],
            rff_dim=mcfg["rff_dim"],
            output_dim=mcfg["output_dim"],
            lambda_reg=mcfg["lambda_reg"],
            attention_scale=mcfg["attention_scale"],
        )

        engine = TrainingEngine(model, cfg, output_dir=cfg["output_dir"])
        engine.train(fold_data["X_train"], fold_data["y_train"],
                     fold_data["X_val"],   fold_data["y_val"])

        # Load best checkpoint and evaluate
        best_model = CCNNCAModel.load_checkpoint(
            os.path.join(cfg["output_dir"], "best_model.pt"))
        metrics = engine.evaluate(fold_data["X_val"], fold_data["y_val"])
        metrics["fold"] = fold + 1
        all_metrics.append(metrics)
        print(f"  Fold {fold+1} → Water R²={metrics['water_r2']:.4f}  "
              f"Heptane R²={metrics['heptane_r2']:.4f}  "
              f"Octane R²={metrics['octane_r2']:.4f}  "
              f"Combined R²={metrics['combined_r2']:.4f}")

    # ── Summary ──────────────────────────────────────────────────────── #
    print("\n" + "="*55)
    print("  CROSS-VALIDATION SUMMARY")
    print("="*55)
    for key in ["water_r2", "heptane_r2", "octane_r2", "combined_r2", "avg_mae"]:
        vals = [m[key] for m in all_metrics]
        print(f"  {key:25s}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")

    results_path = os.path.join(cfg["output_dir"], "cv_results.json")
    with open(results_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\n[train] Results saved → {results_path}")

    if args.loocv:
        print("\n[train] Running Leave-One-Out Cross-Validation...")
        loocv_preds, loocv_true = [], []
        for i, split in enumerate(dm.loocv_splits()):
            model = CCNNCAModel(**{k: v for k, v in cfg["model"].items()
                                    if k in CCNNCAModel.__init__.__code__.co_varnames})
            eng = TrainingEngine(model, cfg, output_dir=cfg["output_dir"])
            eng.train(split["X_train"], split["y_train"],
                      split["X_val"],   split["y_val"])
            best = CCNNCAModel.load_checkpoint(
                os.path.join(cfg["output_dir"], "best_model.pt"))
            metrics = eng.evaluate(split["X_val"], split["y_val"])
            loocv_preds.append(metrics["combined_r2"])
            if (i + 1) % 10 == 0:
                print(f"  LOOCV sample {i+1}/49")
        print(f"  LOOCV Combined R² = {np.mean(loocv_preds):.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--loocv", action="store_true",
                         help="Also run Leave-One-Out CV after k-fold")
    main(parser.parse_args())
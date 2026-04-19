import argparse, os, sys, yaml
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.ccnnca_model import CCNNCAModel
from src.optimization.optimization_engine import OptimizationEngine
from src.data.data_manager import DataManager, generate_synthetic_dataset, PARAM_NAMES


def main(args):
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    dataset_path = cfg["data"]["dataset_path"]
    if not os.path.exists(dataset_path):
        generate_synthetic_dataset(save_path=dataset_path)

    dm = DataManager(dataset_path,
                     feature_cols=cfg["data"]["feature_columns"],
                     target_cols=cfg["data"]["target_columns"])
    _, _, scaler = dm.full_scaled()

    model = CCNNCAModel.load_checkpoint(args.checkpoint)

    bounds_cfg = cfg.get("param_bounds", {})
    param_bounds = [(v[0], v[1]) for v in bounds_cfg.values()] \
        if bounds_cfg else None

    ocfg = cfg.get("optimization", {})
    engine = OptimizationEngine(
        model=model,
        param_bounds=param_bounds,
        weights=ocfg.get("weights"),
        scaler=scaler,
        n_starts=ocfg.get("n_starts", 10),
        max_iter=ocfg.get("max_iter", 1000),
        ftol=ocfg.get("ftol", 1e-9),
    )

    results = engine.optimize_all()
    engine.print_summary(results, PARAM_NAMES)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="outputs/best_model.pt")
    parser.add_argument("--config", default="configs/default.yaml")
    main(parser.parse_args())
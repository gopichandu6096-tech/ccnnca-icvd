import argparse, os, sys, yaml
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.ccnnca_model import CCNNCAModel
from src.interpretability.interpretability_module import InterpretabilityModule
from src.data.data_manager import DataManager, generate_synthetic_dataset


def main(args):
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    dataset_path = cfg["data"]["dataset_path"]
    if not os.path.exists(dataset_path):
        generate_synthetic_dataset(save_path=dataset_path)

    dm = DataManager(dataset_path,
                     feature_cols=cfg["data"]["feature_columns"],
                     target_cols=cfg["data"]["target_columns"])
    X_scaled, _, _ = dm.full_scaled()

    model = CCNNCAModel.load_checkpoint(args.checkpoint)

    explainer = InterpretabilityModule(
        model=model,
        output_dir=cfg.get("output_dir", "outputs/")
    )

    importance = explainer.extract_attention_weights(X_scaled)
    explainer.print_ranking_table(importance)
    explainer.plot_attention_weights(importance)
    print("\n[explain] Feature importance analysis complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="outputs/best_model.pt")
    parser.add_argument("--config", default="configs/default.yaml")
    main(parser.parse_args())
import torch

from scripts.train import train_model
from scripts.evaluate import evaluate


def run_ablation(config):
    """
    Runs ablation configurations based on toggles.
    """

    ablations = {
        "full": {
            "use_kl": True,
            "use_calibration": True,
            "use_attention_alignment": True,
        },
        "no_kl": {
            "use_kl": False,
            "use_calibration": True,
            "use_attention_alignment": True,
        },
        "no_calibration": {
            "use_kl": True,
            "use_calibration": False,
            "use_attention_alignment": True,
        },
        "no_attention_alignment": {
            "use_kl": True,
            "use_calibration": True,
            "use_attention_alignment": False,
        },
    }

    results = {}

    for name, flags in ablations.items():
        print(f"\n===== Running Ablation: {name} =====")

        model_path = train_model(
            config=config,
            use_kl=flags["use_kl"],
            use_calibration=flags["use_calibration"],
            use_attention_alignment=flags["use_attention_alignment"],
        )

        eval_results = evaluate(
            model_path=model_path,
            data_dir=config["test_dir"],
            backbone=config["backbone"],
            num_classes=config["num_classes"]
        )

        results[name] = eval_results

    return results
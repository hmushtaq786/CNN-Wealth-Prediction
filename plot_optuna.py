import joblib
import numpy as np
import matplotlib.pyplot as plt
import optuna
import argparse

def smooth_curve(values, window=3):
    return np.convolve(values, np.ones(window)/window, mode='valid')

def main():
    allowed_indices = ["ndvi", "vari", "msavi", "mndwi", "ndmi", "ndbi"]
    allowed_models = ["resnet", "efficientnet", "vgg"]

    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=str, required=True, choices=allowed_indices,
                    help=f"Index must be one of: {', '.join(allowed_indices)}")
    parser.add_argument("--model", type=str, required=True, choices=allowed_models,
                    help=f"Model must be one of: {', '.join(allowed_models)}")

    args = parser.parse_args()

    storage = f"sqlite:///optuna/storage/single/{args.model}/{args.model}_{args.index}.db"
    study = optuna.load_study(study_name=f"{args.model}_{args.index}", storage=storage)

    scores_path = f"optuna/r2_scores/single/{args.model}/{args.index}/trial"
    print(args.model, args.index)
    plt.figure(figsize=(12, 6))
    for trial in study.trials:
        trial_id = trial.user_attrs.get("slurm_id", trial.number)
        try:
            r2_scores = np.load(f"{scores_path}_{trial_id}.npy")
            r2_scores = np.clip(r2_scores, 0, 1)  # Prevent outlier distortion
            smoothed = smooth_curve(r2_scores, window=2)  # Apply smoothing
            if np.mean(r2_scores) > 0:  # "successful" trial
                plt.plot(smoothed, alpha=0.9, label=f'Trial {trial_id}')
            else:  # faded failed trials
                plt.plot(smoothed, alpha=0.3, linestyle="--")
        except FileNotFoundError:
            print(f"⚠️ Skipping trial {trial_id} — no .npy file found.")

    plt.xlabel("Epoch")
    plt.ylabel("R² Score")
    plt.title("R² per Epoch for Each Trial")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"optuna/plots/single/{args.model}_{args.index}.png")
    plt.close()

    # Save the Optuna study
    # joblib.dump(study, "optuna/studies/resnet34.pkl")
    print("✅ Plot and study saved successfully!")


if __name__ == "__main__":
    main()
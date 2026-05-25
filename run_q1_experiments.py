import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import mlflow

os.environ["MLFLOW_TRACKING_USERNAME"] = "ashos22"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "78b25562699413e85d83414b742443e9df7c5cb5"
mlflow.set_tracking_uri("https://dagshub.com/ashos22/ML_Assignment.mlflow")

from cs231n.classifiers.fc_net import FullyConnectedNet
from cs231n.data_utils import get_CIFAR10_data
from cs231n.solver import Solver

print("Loading data...")
data = get_CIFAR10_data()

N_TRAIN = 500
small_data = {
    "X_train": data["X_train"][:N_TRAIN],
    "y_train": data["y_train"][:N_TRAIN],
    "X_val":   data["X_val"][:200],
    "y_val":   data["y_val"][:200],
}
print("Data ready.")

experiments = [
    {"name": "sgd",          "hidden_dims": [100],      "update_rule": "sgd",          "learning_rate": 1e-3},
    {"name": "sgd-momentum", "hidden_dims": [100],      "update_rule": "sgd_momentum", "learning_rate": 1e-3},
    {"name": "adam",         "hidden_dims": [100],      "update_rule": "adam",         "learning_rate": 1e-3},
    {"name": "adam-deep",    "hidden_dims": [100, 100], "update_rule": "adam",         "learning_rate": 1e-3},
]

mlflow.set_experiment("Q1-FullyConnectedNets")

for cfg in experiments:
    print(f"Training {cfg['name']}...", end=" ", flush=True)

    model = FullyConnectedNet(cfg["hidden_dims"], weight_scale=1e-2)
    solver = Solver(
        model, small_data,
        num_epochs=3,
        batch_size=50,
        update_rule=cfg["update_rule"],
        optim_config={"learning_rate": cfg["learning_rate"]},
        verbose=False,
    )
    solver.train()

    best_val    = float(max(solver.val_acc_history))
    final_train = float(solver.train_acc_history[-1])
    final_loss  = float(solver.loss_history[-1])
    print(f"best_val={best_val:.3f}")

    with mlflow.start_run(run_name=cfg["name"]):
        mlflow.log_params({
            "hidden_dims":   str(cfg["hidden_dims"]),
            "update_rule":   cfg["update_rule"],
            "learning_rate": cfg["learning_rate"],
            "num_epochs":    3,
            "batch_size":    50,
            "n_train":       N_TRAIN,
        })
        mlflow.log_metrics({
            "best_val_acc":    best_val,
            "final_train_acc": final_train,
            "final_loss":      final_loss,
        })

print("Done. Check dagshub.com/ashos22/ML_Assignment -> Experiments")

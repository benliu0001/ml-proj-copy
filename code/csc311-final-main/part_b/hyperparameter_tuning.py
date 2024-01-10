from __future__ import annotations

from .model import MIRT, IRTDataset
from dataclasses import dataclass
from typing import Callable
import random


@dataclass
class Hyperparameters:
    dim: int
    l2_reg_weight: float

    @classmethod
    def random(cls) -> Hyperparameters:
        return cls(
            dim=random.randint(2, 6),
            l2_reg_weight=random.uniform(0, 4),
        )


class HyperparameterTuner:
    model_factory: Callable[[Hyperparameters], MIRT]

    def __init__(self, model_factory: Callable[[Hyperparameters], MIRT]) -> None:
        self.model_factory = model_factory

    def run(self, train: IRTDataset, val: IRTDataset, train_iterations: int) -> None:
        """
        Runs the hyperparameter tuning process.
        """

        best_val_acc = 0
        best_hyperparameters = None

        while True:
            # Generate random hyperparameters
            hyperparameters = Hyperparameters.random()

            # Train model with these hyperparameters
            model = self.model_factory(hyperparameters)
            model.fit_to_dataset(train, val, n_iterations=train_iterations, lr=0.02, verbose=True)

            # Evaluate model on validation set
            val_acc = model.evaluate_accuracy(val)
            print(f"Validation accuracy: {val_acc}")

            # Save model if it's the best so far
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_hyperparameters = hyperparameters

                print(f"New best validation accuracy: {best_val_acc}")
                print(f"New best hyperparameters: {best_hyperparameters}")

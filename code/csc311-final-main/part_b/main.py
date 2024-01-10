from .model import MIRT
from .data import IRTDataset
from .hyperparameter_tuning import HyperparameterTuner, Hyperparameters

import matplotlib.pyplot as plt

train = IRTDataset.from_file("data/train_data.csv")
val = IRTDataset.from_file("data/valid_data.csv")
test = IRTDataset.from_file("data/test_data.csv")


def _model_factory(h: Hyperparameters) -> MIRT:
    return MIRT(train.n_questions(), train.n_users(), dim=h.dim, l2_reg_weight=h.l2_reg_weight)


def tune(train_iterations: int) -> None:
    tuner = HyperparameterTuner(_model_factory)
    tuner.run(train, val, train_iterations)


def main(iterations: int = 50) -> MIRT:
    model = _model_factory(Hyperparameters(dim=3, l2_reg_weight=2.0))
    snapshots = model.fit_to_dataset(train, val, n_iterations=iterations, lr=0.02, verbose=True)

    print(f"Final validation accuracy: {model.evaluate_accuracy(val)}")
    print(f"Final test accuracy: {model.evaluate_accuracy(test)}")

    its = [s.iteration for s in snapshots]
    train_accs = [s.train_acc for s in snapshots]
    val_accs = [s.val_acc for s in snapshots]

    plt.plot(its, train_accs, label="Train")
    plt.plot(its, val_accs, label="Validation")
    plt.xlabel("Iteration")
    plt.ylabel("Accuracy")

    plt.ylim(0.5, 1.0)
    plt.legend()
    plt.show()

    return model

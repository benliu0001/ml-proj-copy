from part_b.data import IRTDataset, IRTDataPoint
from tqdm import trange
import random
import numpy as np


class IRTModel:
    """
    Reimplementation of part_a/item_response.py as an object-oriented model.
    """

    theta: np.ndarray
    beta: np.ndarray

    def __init__(self, n_questions: int, n_users: int) -> None:
        self.theta = np.zeros(n_users)
        self.beta = np.zeros(n_questions)

    def predict_p_correct(self, question: int, user: int) -> float:
        z = self.theta[user] - self.beta[question]

        return np.exp(z) / (1 + np.exp(z))

    def predict_is_correct(self, question: int, user: int) -> bool:
        return self.predict_p_correct(question, user) > 0.5

    def evaluate_accuracy(self, dataset: IRTDataset) -> float:
        correct = 0

        for data_point in dataset.iter():
            c = self.predict_is_correct(data_point.question, data_point.user)
            if c == bool(data_point.correct):
                correct += 1

        return correct / len(dataset)

    def fit_to_dataset(
        self,
        train_set: IRTDataset,
        val_set: IRTDataset,
        *,
        lr: float,
        n_iterations: int,
        verbose: bool = False,
    ) -> None:
        progress = trange(n_iterations, desc="Starting...", disable=not verbose)

        for _ in progress:
            dl_dt = np.zeros_like(self.theta)
            dl_db = np.zeros_like(self.beta)

            for point in train_set.iter():
                z = self.predict_p_correct(point.question, point.user)
                c = point.correct

                dl_dt[point.user] += c - z
                dl_db[point.question] += -(c - z)

            self.theta += lr * dl_dt
            self.beta += lr * dl_db

            progress.set_description(
                f"train_acc={self.evaluate_accuracy(train_set):.3f} val_acc={self.evaluate_accuracy(val_set):.3f}"
            )


class BootstrapSampler:
    dataset: IRTDataset

    def __init__(self, dataset: IRTDataset) -> None:
        self.dataset = dataset

    def sample(self, size: int | None = None) -> IRTDataset:
        """
        Returns a bootstrap sample of the dataset.

        If `size` is not specified, the size of the sample will be the same as
        the size of the dataset.
        """

        n = size or len(self.dataset)

        return IRTDataset.from_data(random.choices(self.dataset._data, k=n))

    def sample_one(self) -> IRTDataPoint:
        """
        Returns a single data point from the dataset.
        """

        return random.choice(self.dataset._data)


class EnsembleModel:
    n_models: int
    models: list[IRTModel]
    train_set: IRTDataset
    sampler: BootstrapSampler

    def __init__(self, n_models: int, train_set: IRTDataset) -> None:
        self.n_models = n_models
        self.train_set = train_set
        self.sampler = BootstrapSampler(train_set)

        self.models = [
            IRTModel(train_set.n_questions(), train_set.n_users()) for _ in range(n_models)
        ]

    def fit(
        self,
        val_set: IRTDataset,
        n_iterations: int,
        lr: float,
        *,
        verbose: bool = False,
    ) -> None:
        """
        Fits the ensemble to the training set.
        """

        for i, model in enumerate(self.models):
            print(f"Fitting model {i + 1} / {self.n_models}:")
            
            model.fit_to_dataset(
                self.sampler.sample(),
                val_set,
                lr=lr,
                n_iterations=n_iterations,
                verbose=verbose,
            )

    def predict_p_correct(self, question: int, user: int) -> float:
        """
        Returns the probability of a user answering a question correctly.
        """

        return sum(model.predict_p_correct(question, user) for model in self.models) / self.n_models

    def evaluate_accuracy(self, dataset: IRTDataset) -> float:
        """
        Returns the accuracy of the model on a dataset.
        """

        correct = 0

        for data_point in dataset.iter():
            c = self.predict_p_correct(data_point.question, data_point.user) > 0.5
            if c == bool(data_point.correct):
                correct += 1

        return correct / len(dataset)


def main() -> None:
    train = IRTDataset.from_file("data/train_data.csv")
    val = IRTDataset.from_file("data/valid_data.csv")
    test = IRTDataset.from_file("data/test_data.csv")

    model = EnsembleModel(3, train)
    model.fit(val, n_iterations=50, lr=0.01, verbose=True)

    print(f"Final validation accuracy: {model.evaluate_accuracy(val)}")
    print(f"Final test accuracy: {model.evaluate_accuracy(test)}")

from __future__ import annotations

import numpy as np
from .data import QuestionIndex, UserIndex, IRTDataset
from .grad import check_grad
from tqdm import trange

from dataclasses import dataclass


@dataclass
class Gradient:
    alphas: np.ndarray
    thetas: np.ndarray
    bias: np.ndarray

    def __add__(self, other: Gradient) -> Gradient:
        return Gradient(
            alphas=self.alphas + other.alphas,
            thetas=self.thetas + other.thetas,
            bias=self.bias + other.bias,
        )

    def __sub__(self, other: Gradient) -> Gradient:
        return Gradient(
            alphas=self.alphas - other.alphas,
            thetas=self.thetas - other.thetas,
            bias=self.bias - other.bias,
        )


@dataclass
class TrainingSnapshot:
    iteration: int
    train_acc: float
    val_acc: float
    train_lld: float


class MIRT:
    """
    Multidimensional Item Response Theory model.

    Adapted from Reckase (2009), Chapter 4.
    """

    alphas: np.ndarray
    thetas: np.ndarray
    bias: np.ndarray

    dim: int

    l2_reg_weight: float

    _elt_parameter_init_bounds: tuple[float, float]

    def __init__(
        self,
        n_questions: int,
        n_users: int,
        dim: int,
        *,
        elt_parameter_init_bounds: tuple[float, float] = (-0.025, 0.025),
        l2_reg_weight: float = 0.12,
    ) -> None:
        """
        Initializes the model with random parameters.

        -   `dim` is the number of latent dimensions.
        -   `elt_parameter_init_bounds` is a tuple `(lo, hi)` specifying the
            bounds of the uniform distribution from which to sample the initial
            values of the model's parameters.
        -   `l2_reg_weight` is the weight of the L2 regularization term, sometimes
            denoted as `λ`.
        """

        self.alphas = np.zeros((n_questions, dim))
        self.thetas = np.zeros((n_users, dim))
        self.bias = np.zeros(n_questions)

        self._elt_parameter_init_bounds = elt_parameter_init_bounds
        self.reinitialize()

        self.dim = dim
        self.l2_reg_weight = l2_reg_weight

    def reinitialize(self) -> None:
        """
        Reinitializes the model with random parameters.
        """

        lo, hi = self._elt_parameter_init_bounds

        self.alphas = np.random.uniform(lo, hi, size=self.alphas.shape)
        self.thetas = np.random.uniform(lo, hi, size=self.thetas.shape)
        self.bias = np.random.uniform(lo, hi, size=self.bias.shape)

    def predict_p_correct(self, question: QuestionIndex, user: UserIndex) -> float:
        """
        Returns the probability that the given user will answer the given
        question correctly.
        """

        alpha = self.alphas[question]
        theta = self.thetas[user]
        d = self.bias[question]

        x = np.dot(alpha, theta) + d

        return 1 / (1 + np.exp(-x))

    def predict_is_correct(self, question: QuestionIndex, user: UserIndex) -> bool:
        """
        Returns whether the model predicts the given user will answer the given
        question correctly. Equivalent to `predict_p_correct(...) > 0.5`.
        """

        return self.predict_p_correct(question, user) > 0.5

    def log_likelihood(self, correct: int, question: QuestionIndex, user: UserIndex) -> float:
        """
        Returns the log-likelihood of the model on the given data point.
        """

        z = self.predict_p_correct(question, user)

        return correct * np.log(z) + (1 - correct) * np.log(1 - z)

    def total_log_likelihood(self, dataset: IRTDataset) -> float:
        """
        Returns the total log-likelihood of the model on the given dataset.
        """

        return sum(
            self.log_likelihood(point.correct, point.question, point.user)
            for point in dataset.iter()
        )

    def evaluate_accuracy(self, dataset: IRTDataset) -> float:
        """
        Returns the accuracy of the model on the given dataset.
        """

        n_correct = 0

        for point in dataset.iter():
            pred = self.predict_is_correct(point.question, point.user)

            if pred == bool(point.correct):
                n_correct += 1

        return n_correct / len(dataset)

    def _compute_regularization_gradient(self) -> Gradient:
        """
        Computes the gradients of the L2 regularization term with respect to the
        parameters of the model.

        Regularization is computed as
        ```
        R = λ * (||a||² + ||θ||² + ||d||²)
        ```
        where `||x||²` is the squared L2 norm of vector `x`, and `λ` is the
        regularization weight.
        ```
        """

        d_thetas = self.l2_reg_weight * 2 * self.thetas
        d_alphas = self.l2_reg_weight * 2 * self.alphas
        d_bias = self.l2_reg_weight * 2 * self.bias

        return Gradient(alphas=d_alphas, thetas=d_thetas, bias=d_bias)

    def _compute_gradient(self, dataset: IRTDataset) -> Gradient:
        """
        Computes the gradients of the log-likelihood with respect to the
        parameters of the model.

        Returns a `Gradient` object consisting of `∂l/∂v` for each parameter `v`, where `l` is the
        log-likelihood of the model on the given dataset.
        """

        d_thetas = np.zeros_like(self.thetas)
        d_alphas = np.zeros_like(self.alphas)
        d_bias = np.zeros_like(self.bias)

        for point in dataset.iter():
            l = self.log_likelihood(point.correct, point.question, point.user)
            p = self.predict_p_correct(point.question, point.user)
            t = point.correct

            # ∂l/∂p - derivative of log-likelihood with respect to the probability of correctness p
            dl_dp = (t / p) - (1 - t) / (1 - p)

            # ∂p/∂z - derivative of sigmoid with respect to its input z
            dp_dz = p * (1 - p)

            for l in range(self.dim):
                # ∂z/∂a_l - derivative of z with respect to alpha_l
                dz_da_l = self.thetas[point.user][l]

                # ∂z/∂t_l - derivative of z with respect to theta_l
                dz_dt_l = self.alphas[point.question][l]

                d_alphas[point.question][l] += dl_dp * dp_dz * dz_da_l
                d_thetas[point.user][l] += dl_dp * dp_dz * dz_dt_l

            # ∂z/∂d - derivative of z with respect to bias d; this is just 1.
            dz_dd = 1
            d_bias[point.question] += dl_dp * dp_dz * dz_dd

        return Gradient(alphas=d_alphas, thetas=d_thetas, bias=d_bias)

    def fit_to_dataset(
        self,
        train: IRTDataset,
        val: IRTDataset,
        *,
        n_iterations: int = 50,
        lr: float = 0.01,
        verbose: bool = True,
    ) -> list[TrainingSnapshot]:
        """
        Fits the model to the given dataset using gradient descent.
        """

        progress = trange(n_iterations, desc="Starting...", disable=not verbose)

        snapshots = []

        for i in progress:
            grad = self._compute_gradient(train) - self._compute_regularization_gradient()

            self.alphas += lr * grad.alphas
            self.thetas += lr * grad.thetas
            self.bias += lr * grad.bias

            snapshot = TrainingSnapshot(
                iteration=i,
                train_acc=self.evaluate_accuracy(train),
                val_acc=self.evaluate_accuracy(val),
                train_lld=self.total_log_likelihood(train),
            )

            progress.set_description(
                f"train_acc={snapshot.train_acc:.3f}, "
                f"val_acc={snapshot.val_acc:.3f}, "
                f"-train_lld={-snapshot.train_lld:.3f}"
            )

            snapshots.append(snapshot)

        return snapshots

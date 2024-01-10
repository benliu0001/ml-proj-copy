from utils import load_train_csv, load_train_sparse, load_valid_csv, load_public_test_csv
from .data import get_data_dir

import numpy as np
from typing import Callable
from tqdm import trange
from dataclasses import dataclass

import matplotlib.pyplot as plt


T = np.ndarray | float


def sigmoid(x: T) -> T:
    """Apply sigmoid function."""
    return np.exp(x) / (1 + np.exp(x))


def p_correct(theta: T, beta: T) -> T:
    """
    Compute the probability of correct response.
    """

    if sigmoid(theta - beta) == 1:
        raise Exception("p_correct is 1")

    return sigmoid(theta - beta)


def neg_log_likelihood(data: dict[str, list], theta: np.ndarray, beta: np.ndarray) -> float:
    """Compute the negative log-likelihood.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    log_lklihood = 0.0

    assert len(data["user_id"]) == len(data["question_id"]) == len(data["is_correct"])

    for idx in range(len(data["user_id"])):
        j = data["question_id"][idx]
        i = data["user_id"][idx]
        c = data["is_correct"][idx]

        p = p_correct(theta[i], beta[j])
        logp = (c * np.log(p)) + (1 - c) * np.log(1 - p)

        log_lklihood += logp

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return -log_lklihood


def log_likelihood(data: dict[str, list], theta: np.ndarray, beta: np.ndarray) -> float:
    return -neg_log_likelihood(data, theta, beta)


def _approximate_log_likelihood_gradient(
    data: dict[str, list],
    theta: np.ndarray,
    beta: np.ndarray,
    epsilon: float = 1e-5,
) -> tuple[np.ndarray, np.ndarray]:
    """
    log_likelihood_gradient, but using finite difference approximation.

    Used for gradient checking. Not used in production, since it is
    very slow.
    """

    dht = np.zeros_like(theta)
    dhb = np.zeros_like(beta)

    for i in trange(len(theta)):
        dt = np.zeros_like(theta)
        dt[i] += epsilon
        y2 = log_likelihood(data, theta + dt, beta)
        y1 = log_likelihood(data, theta - dt, beta)

        dht[i] = (y2 - y1) / (2 * epsilon)

    for j in trange(len(beta)):
        db = np.zeros_like(beta)
        db[j] += epsilon
        y2 = log_likelihood(data, theta, beta + db)
        y1 = log_likelihood(data, theta, beta - db)

        dhb[j] = (y2 - y1) / (2 * epsilon)

    return dht, dhb


def check_grad(
    f: Callable[[np.ndarray], float],
    dfdx: Callable[[np.ndarray], np.ndarray],
    x: np.ndarray,
    *,
    epsilon: float = 1e-5,
) -> float:
    """
    Given a function `f(x)` for a vector-valued input `x` and a
    gradient function `dfdx` representing `∂f/∂x`, return
    the error between `dfdx(x)` and `~dfdx(x)` where `~dfdx` is
    an finite-difference approximation of `∂f/∂x`.
    """

    approx_dfdx = np.zeros_like(x)
    supposed_dfdx = dfdx(x)

    for i in range(x.shape[0]):
        dx = np.zeros_like(x)
        dx[i] = epsilon

        y1 = f(x - dx)
        y2 = f(x + dx)

        approx_dfdx[i] = (y2 - y1) / (2 * epsilon)

    return float(
        np.linalg.norm(approx_dfdx - supposed_dfdx) / np.linalg.norm(approx_dfdx + supposed_dfdx)
    )


def log_likelihood_gradient(
    data: dict[str, list],
    theta: np.ndarray,
    beta: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Let `l(C | theta, beta)` be the log-likelihood of the data `C` given the
    parameters `theta` and `beta`.

    Return `(∂l/∂theta, ∂l/∂beta)`.
    """

    dldtheta = np.zeros_like(theta)
    dldbeta = np.zeros_like(beta)

    for idx in range(len(data["user_id"])):
        j = data["question_id"][idx]
        i = data["user_id"][idx]
        c = data["is_correct"][idx]

        z = p_correct(theta[i], beta[j])

        dldtheta[i] += c - z  # c * theta[i] - z
        dldbeta[j] += -(c - z)

    return dldtheta, dldbeta


def update_theta_beta(
    data: dict[str, list],
    lr: float,
    theta: np.ndarray,
    beta: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Update theta and beta using gradient descent.

    You are using alternating gradient descent. Your update should look:
    for i in iterations ...
        theta <- new_theta
        beta <- new_beta

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param theta: Vector
    :param beta: Vector
    :return: tuple of vectors
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################

    ## Gradient checking; commented out for performance reasons ##

    # a_dldt, a_dldb = _approximate_log_likelihood_gradient(data, theta, beta)
    # d_t = np.linalg.norm(dldt - a_dldt) / np.linalg.norm(dldt + a_dldt)
    # d_b = np.linalg.norm(dldb - a_dldb) / np.linalg.norm(dldb + a_dldb)
    # print(f"theta: {d_t} \t beta: {d_b}")

    dldtheta, dldbeta = log_likelihood_gradient(data, theta, beta)

    theta += lr * dldtheta
    beta += lr * dldbeta

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return theta, beta


@dataclass
class IRTTrainSnapshot:
    theta: np.ndarray
    beta: np.ndarray
    train_score: float
    val_score: float
    train_lld: float
    val_lld: float

    iteration: int


@dataclass
class IRTResults:
    theta_final: np.ndarray
    beta_final: np.ndarray

    train_history: list[IRTTrainSnapshot]


def irt(
    train_data: dict[str, list], val_data: dict[str, list], lr: float, iterations: int
) -> IRTResults:
    """Train IRT model.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param val_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param iterations: int
    :return: (theta, beta, val_acc_lst)
    """

    n_users = max(train_data["user_id"]) + 1  # type: ignore
    n_questions = max(train_data["question_id"]) + 1  # type: ignore

    theta = np.random.uniform(-0.1, 0.1, n_users)
    beta = np.random.uniform(-0.1, 0.1, n_questions)

    history: list[IRTTrainSnapshot] = [
        IRTTrainSnapshot(
            theta=theta.copy(),
            beta=beta.copy(),
            train_score=evaluate(data=train_data, theta=theta, beta=beta),
            val_score=evaluate(data=val_data, theta=theta, beta=beta),
            train_lld=neg_log_likelihood(train_data, theta=theta, beta=beta),
            val_lld=neg_log_likelihood(val_data, theta=theta, beta=beta),
            iteration=0,
        )
    ]

    progress = trange(iterations)

    for i in progress:
        theta, beta = update_theta_beta(train_data, lr, theta, beta)

        train_neg_lld = neg_log_likelihood(train_data, theta=theta, beta=beta)
        val_neg_lld = neg_log_likelihood(val_data, theta=theta, beta=beta)

        train_score = evaluate(data=train_data, theta=theta, beta=beta)
        val_score = evaluate(data=val_data, theta=theta, beta=beta)

        history.append(
            IRTTrainSnapshot(
                theta=theta.copy(),
                beta=beta.copy(),
                train_score=train_score,
                val_score=val_score,
                train_lld=train_neg_lld,
                val_lld=val_neg_lld,
                iteration=i + 1,
            )
        )

        progress.set_description(
            f"train_acc: {train_score:.3f} | val_acc: {val_score:.3f} | train_lld: {train_neg_lld:.3f} | val_lld: {val_neg_lld:.3f}"
        )

    return IRTResults(theta_final=theta, beta_final=beta, train_history=history)


def evaluate(data: dict[str, list], theta: np.ndarray, beta: np.ndarray) -> float:
    """Evaluate the model given data and return the accuracy.
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}

    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    pred = []
    for i, q in enumerate(data["question_id"]):
        u = data["user_id"][i]
        x = (theta[u] - beta[q]).sum()
        p_a = sigmoid(x)

        pred.append(p_a >= 0.5)
    return np.sum((data["is_correct"] == np.array(pred))) / len(data["is_correct"])


def main():
    train_data = load_train_csv(get_data_dir())
    # You may optionally use the sparse matrix.
    sparse_matrix = load_train_sparse(get_data_dir())
    val_data = load_valid_csv(get_data_dir())

    test_data = load_public_test_csv(get_data_dir())

    #####################################################################
    # TODO:                                                             #
    # Tune learning rate and number of iterations. With the implemented #
    # code, report the validation and test accuracy.                    #
    #####################################################################
    lr = 0.01
    iterations = 30

    results = irt(train_data, val_data, lr, iterations)

    print(f"Final snapshot: {results.train_history[-1]}")

    plot_p_correct_wrt_theta(results.beta_final, [0,1,2])
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################

    #####################################################################
    # TODO:                                                             #
    # Implement part (d)                                                #
    #####################################################################
    pass
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


def plot_p_correct_wrt_theta(beta: np.ndarray, js: list[int]) -> None:
    for j in js:
        theta = np.linspace(-4, 4, 100)
        plt.plot(theta, sigmoid(theta - beta[j]), label=f"Question j={j + 1}")
    
    plt.xlabel("Theta")
    plt.ylabel("P(correct)")

    plt.legend()

    plt.show()


def plot_results(
    results: IRTResults,
    train_data: dict[str, list],
    val_data: dict[str, list],
) -> None:
    # Plot the training and validation accuracy w.r.t. iterations

    plt.plot(
        [x.iteration for x in results.train_history],
        [x.train_score for x in results.train_history],
        label="Train",
    )

    plt.plot(
        [x.iteration for x in results.train_history],
        [x.val_score for x in results.train_history],
        label="Validation",
    )

    plt.legend()

    plt.xlabel("Iteration")
    plt.ylabel("Accuracy")

    plt.ylim(0.5, 1.0)

    plt.show()

    # Plot the training and validation log-likelihood w.r.t. iterations

    plt.plot(
        [x.iteration for x in results.train_history],
        [x.train_lld for x in results.train_history],
        label="Train",
    )

    plt.plot(
        [x.iteration for x in results.train_history],
        [x.val_lld for x in results.train_history],
        label="Validation",
    )

    plt.legend()

    plt.xlabel("Iteration")
    plt.ylabel("Negative log-likelihood (lower is better)")

    plt.show()

    # Plot the training and validation log-likelihood w.r.t. iterations
    # normalized by the number of data points

    plt.plot(
        [x.iteration for x in results.train_history],
        [x.train_lld / len(train_data["user_id"]) for x in results.train_history],
        label="Train",
    )

    plt.plot(
        [x.iteration for x in results.train_history],
        [x.val_lld / len(val_data["user_id"]) for x in results.train_history],
        label="Validation",
    )

    plt.legend()

    plt.xlabel("Iteration")
    plt.ylabel("Negative log-likelihood per data point (lower is better)")

    plt.show()


if __name__ == "__main__":
    main()

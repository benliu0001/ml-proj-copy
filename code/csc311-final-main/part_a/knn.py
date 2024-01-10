from sklearn.impute import KNNImputer
from utils import *

import matplotlib.pyplot as plt

import os


def knn_impute_by_user(matrix, valid_data, k, *, report: bool = True) -> float:
    """Fill in the missing values using k-Nearest Neighbors based on
    student similarity. Return the accuracy on valid_data.

    See https://scikit-learn.org/stable/modules/generated/sklearn.impute.KNNImputer.html for details.

    :param matrix: 2D sparse matrix
    :param valid_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :return: float
    """
    nbrs = KNNImputer(n_neighbors=k)
    # We use NaN-Euclidean distance measure.
    mat = nbrs.fit_transform(matrix)
    acc = sparse_matrix_evaluate(valid_data, mat)
    if report:
        print("Validation Accuracy: {}".format(acc))
    return acc


def knn_impute_by_item(matrix, valid_data, k, *, report: bool = True) -> float:
    """Fill in the missing values using k-Nearest Neighbors based on
    question similarity. Return the accuracy on valid_data.

    :param matrix: 2D sparse matrix
    :param valid_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :return: float
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    nbrs = KNNImputer(n_neighbors=k)
    mat = nbrs.fit_transform(matrix.T)
    acc = sparse_matrix_evaluate(valid_data, mat.T)

    if report:
        print("Validation Accuracy: {}".format(acc))

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return acc


def main():
    DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

    sparse_matrix = load_train_sparse(DATA_DIR).toarray()

    val_data = load_valid_csv(DATA_DIR)
    test_data = load_public_test_csv(DATA_DIR)

    #####################################################################
    # TODO:                                                             #
    # Compute the validation accuracy for each k. Then pick k* with     #
    # the best performance and report the test accuracy with the        #
    # chosen k*.                                                        #
    #####################################################################

    k_vals = [1, 6, 11, 16, 21]

    by_user_accs = []
    by_item_accs = []

    for strategy in [knn_impute_by_user, knn_impute_by_item]:
        print("Strategy: {}".format(strategy.__name__))

        # of the form (k, validation_accuracy)
        results: list[tuple[int, float]] = []

        for k in k_vals:
            print("k = {}".format(k))
            val_acc = strategy(sparse_matrix, val_data, k)
            print()

            results.append((k, val_acc))

            if strategy == knn_impute_by_user:
                by_user_accs.append(val_acc)
            else:
                by_item_accs.append(val_acc)

        # find best k, k*, and report test accuracy

        k_star = max(results, key=lambda x: x[1])[0]
        print("k* = {}".format(k_star))

        test_acc = strategy(sparse_matrix, test_data, k_star, report=False)
        print("Test Accuracy: {}".format(test_acc))
        print("\n" * 2)

    # plot validation accuracy vs k

    plt.plot(k_vals, by_user_accs, label="By User")
    plt.plot(k_vals, by_item_accs, label="By Item")
    plt.xlabel("k")
    plt.ylabel("Validation Accuracy")

    plt.legend()

    plt.show()

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()

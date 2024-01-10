from utils import *
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data

import numpy as np
import torch
import matplotlib.pyplot as plt

DEVICE = torch.device("mps")

def load_data(base_path="../data"):
    """ Load the data in PyTorch Tensor.

    :return: (zero_train_matrix, train_data, valid_data, test_data)
        WHERE:
        zero_train_matrix: 2D sparse matrix where missing entries are
        filled with 0.
        train_data: 2D sparse matrix
        valid_data: A dictionary {user_id: list,
        user_id: list, is_correct: list}
        test_data: A dictionary {user_id: list,
        user_id: list, is_correct: list}
    """
    train_matrix = load_train_sparse(base_path).toarray()
    valid_data = load_valid_csv(base_path)
    test_data = load_public_test_csv(base_path)

    zero_train_matrix = train_matrix.copy()
    # Fill in the missing entries to 0.
    zero_train_matrix[np.isnan(train_matrix)] = 0
    # Change to Float Tensor for PyTorch.
    zero_train_matrix = torch.FloatTensor(zero_train_matrix)
    train_matrix = torch.FloatTensor(train_matrix)

    return zero_train_matrix, train_matrix, valid_data, test_data


class AutoEncoder(nn.Module):
    def __init__(self, num_question, k=100):
        """ Initialize a class AutoEncoder.

        :param num_question: int
        :param k: int
        """
        super(AutoEncoder, self).__init__()

        # Define linear functions.
        self.g = nn.Linear(num_question, k)
        self.h = nn.Linear(k, num_question)

    def get_weight_norm(self):
        """ Return ||W^1||^2 + ||W^2||^2.

        :return: float
        """
        g_w_norm = torch.norm(self.g.weight, 2) ** 2
        h_w_norm = torch.norm(self.h.weight, 2) ** 2
        return g_w_norm + h_w_norm

    def forward(self, inputs):
        """ Return a forward pass given inputs.

        :param inputs: user vector.
        :return: user vector.
        """
        #####################################################################
        # TODO:                                                             #
        # Implement the function as described in the docstring.             #
        # Use sigmoid activations for f and g.                              #
        #####################################################################

        # h(W(2)g(W(1)v+b(1))+b(2))

        out = inputs

        sigmoid = nn.Sigmoid()

        self.g(inputs)

        out = sigmoid(self.h(sigmoid(self.g(inputs) + self.g.bias)) + self.h.bias)

        #####################################################################
        #                       END OF YOUR CODE                            #
        #####################################################################
        return out


def train(model, lr, lamb, train_data, zero_train_data, valid_data, num_epoch) -> tuple[int, float, float]:
    """ Train the neural network, where the objective also includes
    a regularizer.

    :param model: Module
    :param lr: float
    :param lamb: float
    :param train_data: 2D FloatTensor
    :param zero_train_data: 2D FloatTensor
    :param valid_data: Dict
    :param num_epoch: int
    :return: None
    """
    # TODO: Add a regularizer to the cost function.

    tuples = []

    # Tell PyTorch you are training the model.
    model.train()

    # Define optimizers and loss function.
    optimizer = optim.SGD(model.parameters(), lr=lr)
    num_student = train_data.shape[0]

    for epoch in range(0, num_epoch):
        train_loss = 0.

        for user_id in range(num_student):
            inputs = Variable(zero_train_data[user_id]).unsqueeze(0).to(DEVICE)
            target = inputs.clone()

            optimizer.zero_grad()
            output = model(inputs)

            # Mask the target to only compute the gradient of valid entries.
            nan_mask = np.isnan(train_data[user_id].unsqueeze(0).numpy())
            target[0][nan_mask] = output[0][nan_mask]

            loss = torch.sum((output - target) ** 2.) + (lamb * 0.5 * model.get_weight_norm())
            loss.backward()

            train_loss += loss.item()
            optimizer.step()

        valid_acc = evaluate(model, zero_train_data, valid_data)
        print("Epoch: {} \tTraining Cost: {:.6f}\t "
              "Valid Acc: {}".format(epoch, train_loss, valid_acc))
        tuples.append((epoch, train_loss, valid_acc))
    return tuples
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


def evaluate(model, train_data: object, valid_data: object) -> float:
    """ Evaluate the valid_data on the current model.

    :param model: Module
    :param train_data: 2D FloatTensor
    :param valid_data: A dictionary {user_id: list,
    question_id: list, is_correct: list}
    :return: float
    """
    # Tell PyTorch you are evaluating the model.
    model.eval()

    total = 0
    correct = 0

    for i, u in enumerate(valid_data["user_id"]):
        inputs = Variable(train_data[u]).unsqueeze(0).to(DEVICE)
        output = model(inputs)

        guess = output[0][valid_data["question_id"][i]].item() >= 0.5
        if guess == valid_data["is_correct"][i]:
            correct += 1
        total += 1
    return correct / float(total)


def main():
    zero_train_matrix, train_matrix, valid_data, test_data = load_data()

    #####################################################################
    # TODO:                                                             #
    # Try out 5 different k and select the best k using the             #
    # validation set.                                                   #
    #####################################################################
    # Set model hyperparameters.
    # Set optimization hyperparameters.

    num_question = train_matrix.shape[1]
    lr = [0.01, 0.02, 0.03, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
    # learning rate

    # PART C
    num_epoch = [100, 50, 25, 10]
    lamb = None
    k = [10, 50, 100, 200, 500]
    # fs = open("training_logs.txt", "a")
    for i in range(5):
        for j in range(3):
            for l in range(4):
                model = AutoEncoder(num_question, k[i])
                model.to(DEVICE)
                train(model, lr[j], lamb, train_matrix, zero_train_matrix, valid_data, num_epoch[l])
                # valid_acc = evaluate(model, zero_train_matrix, valid_data)

                # fs.write("Learning Rate "+ str(lr[j]) + "Number of Epochs " + str(num_epoch[l]) + "K " + str(k[i]) + "Accuracy " + str(valid_acc) + "\n")

    # fs.close()

    # PART D

    k_star = 50
    chosen_lr = 0.01
    chosen_iterations = 50
    model = AutoEncoder(num_question, k_star)
    tuples = train(model, chosen_lr, lamb, train_matrix, zero_train_matrix, valid_data,
                   chosen_iterations)

    print("Test Accuracy: " + str(evaluate(model, zero_train_matrix, test_data)))

    x_values = [x[0] for x in tuples]
    loss_values = [x[1] for x in tuples]
    validation_accuracy_values = [x[2] for x in tuples]
    plt.plot(x_values, loss_values)
    plt.xlabel('Epochs')
    plt.ylabel('Training Loss')

    # Show the plot
    plt.show()
    plt.plot(x_values, validation_accuracy_values)
    plt.xlabel('Epochs')
    plt.ylabel('Validation Accuracy')
    plt.show()

    lambdas = [0.001, 0.01, 0.1, 1]
    for a in range(4):
        model = AutoEncoder(num_question, k_star)
        train(model, chosen_lr, lambdas[a], train_matrix, zero_train_matrix, valid_data,
              chosen_iterations)
        print(lambdas[a])

    # PART E

    model = AutoEncoder(num_question, k_star)
    tuples = train(model, chosen_lr, 0.001, train_matrix, zero_train_matrix, valid_data,
                   chosen_iterations)
    print("Test Accuracy: " + str(evaluate(model, zero_train_matrix, test_data)))

    x_values = [x[0] for x in tuples]
    loss_values = [x[1] for x in tuples]
    validation_accuracy_values = [x[2] for x in tuples]
    plt.plot(x_values, loss_values)
    plt.xlabel('Epochs')
    plt.ylabel('Training Loss')

    # Show the plot
    plt.show()
    plt.plot(x_values, validation_accuracy_values)
    plt.xlabel('Epochs')
    plt.ylabel('Validation Accuracy')
    plt.show()
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()

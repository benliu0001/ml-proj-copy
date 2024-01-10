from .model import MIRT
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import numpy as np
from tqdm import trange

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))

def plot_mirt_fixed_theta(
    alpha: np.ndarray,
    model: MIRT,
    *,
    resolution: int = 100,
    bounds: float = 3,
) -> None:
    """
    Plots the probability of answering a question correctly as a function of
    the latent ability parameter, for a fixed question.
    """

    # a = np.linspace(-5, 5, 25) 
    # b = np.linspace(-5, 5, 25) 
    # x, y = np.meshgrid(a, b) 
    # z = np.cos(np.sqrt(x**2 + y**2)) 

    # fig = plt.figure() 
    # wf = fig.add_subplot(projection='3d') 
    # wf.plot_wireframe(x, y, z, color ='blue') 

    # wf.set_title('Example 2') 
    # plt.show() 

    if model.dim != 2:
        raise ValueError("Can only plot for 2-dimensional models.")

    a = np.linspace(-bounds, bounds, resolution)
    b = np.linspace(-bounds, bounds, resolution)
    x, y = np.meshgrid(a, b)
    z = np.zeros_like(x)

    for i in trange(resolution):
        for j in range(resolution):
            z[i, j] = sigmoid(np.dot(alpha, np.array([x[i, j], y[i, j]])))

    fig = plt.figure()

    ax = fig.add_subplot(projection="3d")
    ax.plot_wireframe(x, y, z, color="blue", rstride=6, cstride=6)

    # Label axes

    ax.set_xlabel(r"$\theta_{i1}$")
    ax.set_ylabel(r"$\theta_{i2}$")
    ax.set_zlabel(r"$P(correct)$")

    plt.show()
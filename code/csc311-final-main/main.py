import sys

from part_a.knn import main as knn_main
from part_a.item_response import main as item_response_main
from part_a.ensemble import main as ensemble_main
from part_b.main import tune
from part_b.main import main as part_b_main
from part_b.visualization import plot_mirt_fixed_theta

import numpy as np


if __name__ == "__main__":
    USAGE = "Usage: python main.py <part>. See README.md for more details."

    if len(sys.argv) < 2:
        print(USAGE)
        sys.exit(1)
    if sys.argv[1] == "knn":
        knn_main()
    elif sys.argv[1] == "irt":
        item_response_main()
    elif sys.argv[1] == "ensemble":
        ensemble_main()
    elif sys.argv[1] == "part_b":
        part_b_main()
    elif sys.argv[1] == "part_b::plot":
        model = part_b_main(2)
        plot_mirt_fixed_theta(np.array([5, 1]), model)
    elif sys.argv[1] == "part_b::tune":
        tune(50)
    else:
        print(f"Unknown part: {sys.argv[1]}")
        print(USAGE)
        sys.exit(1)
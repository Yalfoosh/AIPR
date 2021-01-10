from typing import Any, Dict

from matplotlib import pyplot as plt
import numpy as np

from .constants import CLASS_NAME_TO_NAME, SUB


def get_state_over_time_graphs(result_dict: Dict[str, Dict[str, Any]]):
    fig, ax = plt.subplots(3, 2, figsize=(16, 18))

    for n, (class_name, title) in enumerate(CLASS_NAME_TO_NAME.items()):
        i = n // 2
        j = n % 2

        current_results = result_dict[class_name]
        x_axis = [x["t"] for x in current_results["states"]]
        y_axis = np.array([x["x"] for x in current_results["states"]])
        y_axis = y_axis.reshape(len(y_axis), -1).T

        ax[i][j].set_title(title)
        ax[i][j].set_xlabel("t")
        ax[i][j].set_ylabel("x")

        for index, y in enumerate(y_axis):
            ax[i][j].plot(x_axis, y, label=f"x{str(index).translate(SUB)}")

        ax[i][j].legend()

    return fig, ax


def print_diffs(diff_dict: Dict[str, float], n_decimals: int = 3):
    for key, diff in diff_dict.items():
        print(f"{CLASS_NAME_TO_NAME[key]}:")

        for i, d in enumerate(diff.reshape(-1)):
            print(f"\tΔx{str(i).translate(SUB)} = {d:.0{n_decimals}f}")

        print()

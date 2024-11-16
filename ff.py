import ast

import matplotlib.pyplot as plt  # type: ignore
import numpy as np  # type: ignore


def read_boolean_file(file_path):
    with open(file_path, "r") as file:
        lines = file.readlines()
    data = []
    for line in lines:
        _, boolean_values = line.split(",", 1)
        boolean_list = ast.literal_eval(boolean_values.strip())
        int_list = [int(value) for value in boolean_list]
        data.append(int_list)
    data_array = np.array(data, dtype=int)
    return data_array


def parse_data(lines):
    data_arrays = []
    for line in lines:
        if line.strip():  # This checks if the line is not empty
            array_str = line.split(",", 1)[1].strip().strip("[]")
            array = np.array(list(map(float, array_str.split(","))))
            data_arrays.append(array)
    return np.array(data_arrays)


def read_data(filename):
    data = []
    with open(filename, "r") as file:
        for line in file:
            value = line.strip().split(", ")[1]
            data.append(float(value))
    return np.array(data, dtype=object)


def plotLens(array):
    plt.figure(figsize=(10, 5))
    plt.step(range(len(array)), array, where="post")
    plt.show()

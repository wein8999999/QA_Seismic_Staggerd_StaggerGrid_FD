# !/usr/bin/env python
# -*-coding:utf-8 -*-
# @Time    : 2023/3/13 17:38
# @Author  : Wen Xixiang
# @Version : python3.9
import numpy as np
import scipy as sp
import networkx as nx
import sys
from typing import Any, Literal, Tuple, Union, List, Iterable
from scipy.signal import fftconvolve
import warnings

from matplotlib import pyplot as plt


def cholesky_solver(A, b):
    cho, low = sp.linalg.cho_factor(A)
    solution = sp.linalg.cho_solve((cho, low), b)
    return solution


def to_fixed_point(binary: np.ndarray, bit_value: np.ndarray) -> np.ndarray:
    """Convert a binary array to a floating-point array represented by bit values.

    Parameters
    ----------
    binary : np.ndarray
        The bit string obtained after sampling.
    bit_value : np.ndarray
        An array consisted by the value of each bit.

    Examples
    --------
    >>> binary = np.array([1,0,1,0,0,1,0,1,0,1,1,0,1,1,1])
    >>> bit_value = np.array([2,1,0.5,0.25,0.125])
    >>> to_fixed_point(binary,bit_value)
    >>> array([2.5  , 2.625, 2.875])
    """

    if not isinstance(binary, np.ndarray):
        raise TypeError(
            f"`binary` must be the instance of np.ndarray, not {type(binary)}"
        )
    if not isinstance(bit_value, np.ndarray):
        raise TypeError(
            f"`bit_value` must be the instance of np.ndarray, not {type(bit_value)}"
        )

    binary, bit_value = binary.flatten(), bit_value.flatten()

    # If sampling model is ISING, change the {-1,1} to {0,1}
    # binary = np.array(binary + np.abs(binary)) / 2

    num_binary_entry = len(binary)
    num_bits = len(bit_value)
    num_x_entry = num_binary_entry // num_bits
    if num_x_entry * num_bits != num_binary_entry:
        raise ValueError("The length of q or bit_value is incorrect.")
    float_array = np.array(
        [
            bit_value @ binary[i * num_bits : (i + 1) * num_bits]
            for i in range(num_x_entry)
        ]
    )
    return float_array.reshape(-1, 1)


def b2f(
    binary: np.ndarray, low_value: Union[int, float], high_value: Union[int, float]
) -> np.ndarray:
    """Convert a binary array to a floating-point array represented by two values.

    Parameters
    ----------
    binary : np.ndarray
        Binary array.
    low_value : Union[int, float]
        Low value.
    high_value : Union[int, float]
        High value.

    Returns
    -------
    float_array : np.ndarray
        Floating-point array.
    """

    if low_value > high_value:
        low_value, high_value = high_value, low_value
    elif low_value == high_value:
        warnings.warn("`low_value` and `high_value` are the same.")

    # If sampling model is ISING, change the {-1,1} to {0,1}
    binary = np.array(binary + np.abs(binary)) / 2
    float_array = np.array(binary.flatten() * high_value - (binary - 1) * low_value)

    return float_array


class Logger(object):
    def __init__(self, filename="default.log", stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        # Flush both streams to ensure all content is written
        self.terminal.flush()
        self.log.flush()

    def close(self):
        # Close the log file when done
        self.log.close()


def show_obj_curve(obj_set):
    fig, ax = plt.subplots(1, 1, figsize=(5, 7))
    ax.plot([i for i in obj_set[1:]], "o-", color="crimson")
    ax.set_ylabel("Misfit Function")
    ax.set_xlabel("Iteration")
    ax.set_yscale("log")
    ax.tick_params(direction="in", length=3, width=1)
    # ax.legend(loc=1, fontsize='small')
    plt.pause(3.0)
    plt.close(fig)

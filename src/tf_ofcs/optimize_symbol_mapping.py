# -*- coding: utf-8
# Copyright (C) 2025 Ivan Konovalov

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import numpy as np
import numpy.typing as npt
import random
import operator
from collections.abc import Callable
import numba

__all__ = ["optimize_symbol_mapping"]


@numba.njit
def _hamming_distance(x: int, y: int) -> int:
    xor = x ^ y
    distance = 0

    while xor != 0:
        distance += xor & 1
        xor >>= 1

    return distance


@numba.njit
def _make_hamming_matrix(M: int) -> npt.NDArray:
    hamming_matrix = np.zeros((M, M), dtype=np.int16)

    for i in range(M):
        for j in range(M):
            hamming_matrix[i, j] = _hamming_distance(i, j)

    return hamming_matrix


@numba.njit
def _errors_from_confusion_matrix(
    mapping: npt.NDArray,
    confusion_matrix: npt.NDArray,
    hamming_matrix: npt.NDArray
) -> int:
    total_errors = 0
    M = len(mapping)

    for i in range(M):
        mapped_i = mapping[i]

        for j in range(M):
            mapped_j = mapping[j]
            total_errors += confusion_matrix[i, j] * hamming_matrix[mapped_i, mapped_j]

    return total_errors


@numba.njit
def _random_swap(x: npt.NDArray, n: int = 1) -> None:
    for i in range(n):
        i = random.randint(0, len(x) - 1)
        j = random.randint(0, len(x) - 1)
        x[i], x[j] = x[j], x[i]


def optimize_symbol_mapping(
    confusion_matrix: npt.NDArray,
    initial_mapping: npt.NDArray,
    iteration_callback: Callable[[int, list[tuple[int, npt.NDArray]]], bool],
    population_size: int = 10000,
    max_survivors: int = 9000,
) -> npt.NDArray:
    """
        Optimizes a symbol mapping to minimize the bit error rate given a confusion
        matrix. Optimization is performed using a variant of genetic algorithm.
        It starts with an initial mapping, then generates a population of candidate
        mappings by randomly swapping elements in the initial mapping. The
        candidates are then evaluated based on their bit error rate, and the best
        candidates are selected to form the next generation. This process is
        repeated until the iteration callback returns false.

        :param confusion_matrix: `np.NDArray`, a square matrix where element
            (i, j) represents the number of times symbol i being classified as
            symbol j.
        :param initial_mapping: `np.NDArray`, an initial mapping of symbols to integers.
        :param iteration_callback: `Callable[[int, list[tuple[int, np.NDArray]]], bool]`,
            a function that is called after each iteration of the optimization.
            It takes the iteration number and a list of (error, mapping) tuples
            as arguments, and returns a boolean value indicating whether the
            optimization should continue (`True` to continue).
        :param population_size: `int`, the number of candidate mappings to generate
            in each iteration.
        :param max_survivors: `int`, the number of best candidate mappings to keep after
            each iteration.

        :returns: `np.NDArray`, best mapping found by the algorithm
    """

    M = len(confusion_matrix)

    hamming_matrix = _make_hamming_matrix(M)

    candidates = [(
        _errors_from_confusion_matrix(initial_mapping, confusion_matrix, hamming_matrix),
        np.copy(initial_mapping)
    )]

    key_function = operator.itemgetter(0)
    iteration = 0

    while True:
        if not iteration_callback(iteration, candidates):
            break

        n_survivors = len(candidates)

        while len(candidates) < population_size:
            candidate_mapping = np.copy(candidates[random.randint(0, n_survivors - 1)][1])
            _random_swap(candidate_mapping)

            metric = _errors_from_confusion_matrix(candidate_mapping, confusion_matrix, hamming_matrix)
            candidates.append((metric, candidate_mapping))

        candidates.sort(key=key_function)
        candidates = candidates[:max_survivors]

        iteration += 1

    return candidates[0][1]

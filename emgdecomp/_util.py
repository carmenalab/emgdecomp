import collections
from typing import Union, List, Tuple, TypeVar, Dict, Set

import numba
import numpy as np


def minimum_distances(
        left_arr: Union[np.ndarray, List],
        right_arr: Union[np.ndarray, List],
        return_indices: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Finds the minimum distance between each element of <left_arr> and any element of <right_arr>. Returns a value for
    element of left_arr that is the shortest distance.

    In numpy parlance, this method is equivalent to
        np.min(np.abs(left_arr[:, np.newaxis] - right_arr), axis=1)
    but without the O(n^2) space/time.
    :param left_arr:
    :param right_arr:
    :return:
    """
    left_arr = np.array(left_arr)
    right_arr = np.array(right_arr)

    ret = np.empty((len(left_arr),), dtype=left_arr.dtype)
    ret_indices = np.empty((len(left_arr)), dtype=np.int)
    unsorted_ret = np.empty(ret.shape, dtype=ret.dtype)
    unsorted_ret_indices = np.empty(ret_indices.shape, dtype=ret_indices.dtype)
    _minimum_distances_jit(
        left_arr=left_arr,
        right_arr=right_arr,
        unsorted_ret=unsorted_ret,
        unsorted_ret_indices=unsorted_ret_indices,
        ret=ret,
        ret_indices=ret_indices)
    if return_indices:
        return unsorted_ret, unsorted_ret_indices
    else:
        return unsorted_ret


@numba.jit(nopython=True, cache=True)
def _minimum_distances_jit(
        left_arr: Union[np.ndarray, List],
        right_arr: Union[np.ndarray, List],
        ret: np.ndarray,
        ret_indices: np.ndarray,
        unsorted_ret: np.ndarray,
        unsorted_ret_indices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Finds the minimum distance between each element of <left_arr> and any element of <right_arr>. Returns a value for
    element of left_arr that is the shortest distance.

    In numpy parlance, this method is equivalent to
        np.min(np.abs(left_arr[:, np.newaxis] - right_arr), axis=1)
    but without the O(n^2) space/time.
    :param left_arr:
    :param right_arr:
    :return:
    """
    sorted_left_indices = np.argsort(left_arr)
    sorted_left = left_arr[sorted_left_indices]
    sorted_right = np.sort(right_arr)

    right_idx = 0
    for left_idx in range(len(sorted_left)):
        # [ 1, 2, 3, 7,  14 ]
        # [ 4, 6, 7, 10, 15 ]
        left = sorted_left[left_idx]
        right = sorted_right[right_idx]
        if right_idx == len(sorted_right) - 1:
            # No other choice - we're at the end of right
            ret[left_idx] = abs(right - left)
            ret_indices[left_idx] = right_idx
            continue

        while right_idx < len(sorted_right) - 1:
            distance_current = abs(left - sorted_right[right_idx])
            distance_next = abs(left - sorted_right[right_idx + 1])
            if distance_current < distance_next:
                break
            # We got closer; keep going
            right_idx += 1

        ret[left_idx] = abs(sorted_right[right_idx] - left)
        ret_indices[left_idx] = right_idx

    # "Unsort" it to revert it to the original ordering
    for idx in range(len(unsorted_ret)):
        unsorted_ret[sorted_left_indices[idx]] = ret[idx]
        unsorted_ret_indices[sorted_left_indices[idx]] = ret_indices[idx]

    return unsorted_ret, unsorted_ret_indices


T = TypeVar('T')

def find_disconnected_subgraphs(edges: Dict[T, Set[T]]) -> List[List[T]]:
    """
    Finds all disconnected subgraphs within a graph defined by the edges given. Returns one list per disconnected
    subgraph where each list contains the nodes in that subgraph, in sorted order.
    """

    unvisited_nodes = set()
    bidirectional_edges = collections.defaultdict(lambda: set())
    for pre, posts in edges.items():
        unvisited_nodes.add(pre)
        for post in posts:
            unvisited_nodes.add(post)
            bidirectional_edges[pre].add(post)
            bidirectional_edges[post].add(pre)

    ret = []

    # Find all connected subgraphs by doing a BFS with bidirectional (i.e. undirected) edges
    while len(unvisited_nodes) > 0:
        nodes_in_graph = set()
        nodes_to_visit = [next(iter(unvisited_nodes))]
        while len(nodes_to_visit) > 0:
            current_node = nodes_to_visit.pop(0)
            nodes_in_graph.add(current_node)
            if current_node in unvisited_nodes:
                unvisited_nodes.remove(current_node)
            else:
                continue
            if current_node in bidirectional_edges:
                for next_node in bidirectional_edges[current_node]:
                    nodes_to_visit.append(next_node)

        ret.append(sorted(nodes_in_graph))
    return sorted(ret)

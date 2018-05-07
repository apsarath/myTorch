"""
List of utility functions to support various net2net operations (http://arxiv.org/abs/1511.05641).
Base implementation (with support for Python3) is maintained at https://github.com/shagunsodhani/net2net.
Based on a version forked from https://github.com/paengs/Net2Net
Notes:
    * All the functions take as input a numpy array and return as output a numpy array
    * All the functions are stateless so they have to be provided the `indices_to_copy` and `replication_factor`
    explicitly.

"""
import numpy as np

def make_h_wider(teacher_b, indices_to_copy):
    """
    Method to make h wider
    Args:
        teacher_b: weights corresponding to h that needs to be expanded
        indices_to_copy: indices that are to be copied in h
    """
    student_b = teacher_b.copy()
    for i in range(len(indices_to_copy)):
        teacher_index = indices_to_copy[i]
        student_b = np.append(student_b, teacher_b[:, teacher_index][:, np.newaxis], axis=1)
    return student_b


def make_weight_wider_at_input(teacher_w, indices_to_copy, replication_factor):
    """
        Method to make weight wider at the input
        Args:
            teacher_w: weights that needs to be expanded
            indices_to_copy: indices that are to be copied in teacher_w
            replication_factor: factor to normalise the copied weights
        """
    student_w = teacher_w.copy()
    for i in range(len(indices_to_copy)):
        teacher_index = indices_to_copy[i]
        factor = replication_factor[teacher_index] + 1
        assert factor > 1, 'Error in Net2Wider'
        new_weight = teacher_w[teacher_index, :] * (1. / factor)
        new_weight = new_weight[np.newaxis, :]
        student_w = np.concatenate((student_w, new_weight), axis=0)
        student_w[teacher_index, :] = new_weight
    return student_w


def make_bias_wider(teacher_b, indices_to_copy):
    """
        Method to make bias wider
        Args:
            teacher_b: weights corresponding to bias that needs to be expanded
            indices_to_copy: indices that are to be copied in bias
        """
    student_b = teacher_b.copy()
    for i in range(len(indices_to_copy)):
        teacher_index = indices_to_copy[i]
        student_b = np.append(student_b, teacher_b[teacher_index])
    return student_b


def make_weight_wider_at_output(teacher_w, indices_to_copy):
    """
        Method to make weight wider at the output
        Args:
            teacher_w: weights that needs to be expanded
            indices_to_copy: indices that are to be copied in teacher_w
        """
    student_w = teacher_w.copy()
    for i in range(len(indices_to_copy)):
        teacher_index = indices_to_copy[i]
        new_weight = teacher_w[:, teacher_index]
        new_weight = new_weight[:, np.newaxis]
        student_w = np.concatenate((student_w, new_weight), axis=1)
    return student_w

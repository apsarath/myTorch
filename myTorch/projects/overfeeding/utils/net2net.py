"""
List of utility functions to support various net2net operations (http://arxiv.org/abs/1511.05641).
Base implementation (with support for Python3) is maintained at https://github.com/shagunsodhani/net2net.
Based on a version forked from https://github.com/paengs/Net2Net
Notes:
    * All the functions take as input a numpy array and return as output a numpy array
    * All the functions are stateless so they have to be provided the `indices_to_copy` and `replication_factor`
    explicitly.

"""
from copy import deepcopy

import numpy as np


default_val_use_random_noise = None
default_val_use_noise = None


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


def make_bias_wider(teacher_b, indices_to_copy, replication_factor):
    """
        Method to make bias wider
        Args:
            teacher_b: weights corresponding to bias that needs to be expanded
            indices_to_copy: indices that are to be copied in bias
        """
    # noise = generate_noise(replication_factor + 1, size=teacher_b.shape[0])
    student_b = teacher_b.copy()
    for i in range(len(indices_to_copy)):
        teacher_index = indices_to_copy[i]
        student_b = np.append(student_b, teacher_b[teacher_index])
    return student_b


def make_weight_wider_at_output(teacher_w, indices_to_copy, replication_factor, use_noise=default_val_use_noise,
                                use_random_noise=default_val_use_random_noise):
    """
        Method to make weight wider at the output
        Args:
            teacher_w: weights that needs to be expanded
            indices_to_copy: indices that are to be copied in teacher_w
        """

    noise = generate_noise_for_output(replication_factor + 1, size=teacher_w.shape[0],
                                      use_noise=use_noise,
                                      use_random_noise=use_random_noise, teacher_w=teacher_w)

    # print(noise)
    student_w = teacher_w.copy()
    for i in range(len(indices_to_copy)):
        teacher_index = indices_to_copy[i]
        noise_to_add = noise[teacher_index].pop()
        new_weight = teacher_w[:, teacher_index] + np.asarray(noise_to_add)
        new_weight = new_weight[:, np.newaxis]
        student_w = np.concatenate((student_w, new_weight), axis=1)
    for teacher_index in range(teacher_w.shape[1]):
        noise_to_add = noise[teacher_index].pop()
        student_w[:, teacher_index] = teacher_w[:, teacher_index] + noise_to_add
    return student_w.astype(np.float32)


def generate_noise_for_input(replication_factor, size, use_noise=default_val_use_noise,
                             use_random_noise=default_val_use_random_noise):
    """
    Method to generate a list of random matrix where each value if between 0 and 1.
    The size of the matrix is size(replication_factor) x size
    The sum of numbers along any column is 0
    :param replication_factor:
    :return:
    """
    noise_matrix = []
    for val in replication_factor:
        noise_matrix.append(generate_noise_matrix_for_input(a=val, b=size,
                                                            use_noise=use_noise, use_random_noise=use_random_noise))
    return noise_matrix


def generate_noise_for_output(replication_factor, size,
                              use_noise=default_val_use_noise,
                              use_random_noise=default_val_use_random_noise,
                              teacher_w=None):
    """
    Method to generate a list of random matrix where each value if between 0 and 1.
    The size of the matrix is size(replication_factor) x size
    The sum of numbers along any column is 0
    :param replication_factor:
    :return:
    """
    noise_matrix = []
    for val in replication_factor:
        noise_matrix.append(generate_noise_matrix_for_output(a=val, b=size,
                                                             use_noise=use_noise,
                                                             use_random_noise=use_random_noise, weights=teacher_w[val]))
    return noise_matrix


def generate_noise_vector(vector_size, use_noise, use_random_noise=default_val_use_random_noise):
    """
    Method to generate a noise vector of size vector_size.
    :param use_noise:
    :param use_random_noise:
    :param vector_size:
    :return:
    """
    if (vector_size < 1):
        return np.asarray([])
    if (use_noise == False):
        return np.zeros(vector_size)
    if (use_random_noise):
        return np.random.normal(size=vector_size, scale=0.1)
        # return np.random.normal(low=-1, high=1, size=vector_size)
    else:
        rand_vec = np.random.uniform(size=vector_size - 1)
        sorted_vec = [0] + list(np.sort(rand_vec)) + [1]
        vec = np.ediff1d(sorted_vec)
        vec = vec - 1.0 / vector_size
        return vec


def generate_noise_matrix_for_input(a, b, use_noise=default_val_use_noise,
                                    use_random_noise=default_val_use_random_noise):
    tup = []
    for _ in range(b):
        tup.append(generate_noise_vector(a, use_noise, use_random_noise))
    return np.column_stack(tup).tolist()


def generate_noise_matrix_for_output(a, b, use_noise=default_val_use_noise,
                                     use_random_noise=default_val_use_random_noise, weights=None):
    tup = []
    for _ in range(b):
        tup.append(generate_noise_vector(a, use_noise, use_random_noise))
    return np.column_stack(tup).tolist()


if __name__ == "__main__":
    teacher_w1 = np.asarray([[1, 3, 5],
                             [2, 4, 6]])
    teacher_w2 = np.expand_dims(np.asarray([1, 2, 3]), axis=1)
    print(teacher_w2.shape)
    indices_to_copy = np.random.randint(3, size=100)
    replication_factor = np.bincount(indices_to_copy, minlength=3)
    print(indices_to_copy)
    print(replication_factor)
    student_w1 = make_weight_wider_at_output(teacher_w1, indices_to_copy=indices_to_copy,
                                             replication_factor=replication_factor)
    student_w2 = make_weight_wider_at_input(teacher_w2, indices_to_copy=indices_to_copy,
                                            replication_factor=replication_factor)
    print(student_w1)
    print(student_w2)
    a = np.expand_dims(np.asarray([7, 2]), axis=0)
    print(a @ student_w1 @ student_w2)
    print(a @ teacher_w1 @ teacher_w2)
    print(a.shape)

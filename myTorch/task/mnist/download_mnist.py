import os
import functools
import operator
import gzip
import struct
import array
import tempfile
from urllib.request import urlretrieve
from urllib.parse import urljoin
import numpy as np


# the url can be changed by the users of the library (not a constant)
datasets_url = 'http://yann.lecun.com/exdb/mnist/'


class IdxDecodeError(ValueError):
    """Raised when an invalid idx file is parsed."""
    pass


def download_file(fname, target_dir=None, force=False):
    """Download fname from the datasets_url, and save it to target_dir,
    unless the file already exists, and force is False.

    Args:
        fname: str, Name of the file to download
        target_dir: str, Directory where to store the file
        force: bool, Force downloading the file, if it already exists
    Returns:
        fname: str, Full path of the downloaded file
    """
    if not target_dir:
        target_dir = tempfile.gettempdir()
    target_fname = os.path.join(target_dir, fname)

    if force or not os.path.isfile(target_fname):
        url = urljoin(datasets_url, fname)
        urlretrieve(url, target_fname)

    return target_fname


def parse_idx(fd):
    """Parse an IDX file, and return it as a numpy array.

    Args:
        fd: file, File descriptor of the IDX file to parse
    Returns:
        data: numpy.ndarray, Numpy array with the dimensions and the data in the IDX file
    """
    DATA_TYPES = {0x08: 'B',  # unsigned byte
                  0x09: 'b',  # signed byte
                  0x0b: 'h',  # short (2 bytes)
                  0x0c: 'i',  # int (4 bytes)
                  0x0d: 'f',  # float (4 bytes)
                  0x0e: 'd'}  # double (8 bytes)

    header = fd.read(4)
    if len(header) != 4:
        raise IdxDecodeError('Invalid IDX file, file empty or does not contain a full header.')

    zeros, data_type, num_dimensions = struct.unpack('>HBB', header)

    if zeros != 0:
        raise IdxDecodeError('Invalid IDX file, file must start with two zero bytes. '
                             'Found 0x%02x' % zeros)

    try:
        data_type = DATA_TYPES[data_type]
    except KeyError:
        raise IdxDecodeError('Unknown data type 0x%02x in IDX file' % data_type)

    dimension_sizes = struct.unpack('>' + 'I' * num_dimensions,
                                    fd.read(4 * num_dimensions))

    data = array.array(data_type, fd.read())
    data.byteswap()  # looks like array.array reads data as little endian

    expected_items = functools.reduce(operator.mul, dimension_sizes)
    if len(data) != expected_items:
        raise IdxDecodeError('IDX file has wrong number of items. '
                             'Expected: %d. Found: %d' % (expected_items, len(data)))

    return np.array(data).reshape(dimension_sizes)


def download_and_parse_mnist_file(fname, target_dir=None, force=False):
    """Download the IDX file named fname from the URL specified in dataset_url
    and return it as a numpy array.

    Args:
        fname : str, File name to download and parse
        target_dir : str, Directory where to store the file
        force : bool, Force downloading the file, if it already exists

    Returns:
        data : numpy.ndarray, Numpy array with the dimensions and the data in the IDX file
    """
    fname = download_file(fname, target_dir=target_dir, force=force)
    fopen = gzip.open if os.path.splitext(fname)[1] == '.gz' else open
    with fopen(fname, 'rb') as fd:
        return parse_idx(fd)


def download_mnist(target_dir):
    """Download and process the mnist dataset and store it in the target dir.

    Args:
        target_dir: str, Directory where to store the file
    """

    flag_file = os.path.join(target_dir, "flag.p")
    if os.path.isfile(flag_file):
        return

    train_x = download_and_parse_mnist_file('train-images-idx3-ubyte.gz', target_dir)
    target_fname = os.path.join(target_dir, "train_x")
    np.save(target_fname, train_x[0:50000])
    target_fname = os.path.join(target_dir, "valid_x")
    np.save(target_fname, train_x[-10000:])

    train_y = download_and_parse_mnist_file('train-labels-idx1-ubyte.gz', target_dir)
    target_fname = os.path.join(target_dir, "train_y")
    np.save(target_fname, train_y[0:50000])
    target_fname = os.path.join(target_dir, "valid_y")
    np.save(target_fname, train_y[-10000:])

    test_x = download_and_parse_mnist_file('t10k-images-idx3-ubyte.gz', target_dir)
    target_fname = os.path.join(target_dir, "test_x")
    np.save(target_fname, test_x)

    test_y = download_and_parse_mnist_file('t10k-labels-idx1-ubyte.gz', target_dir)
    target_fname = os.path.join(target_dir, "test_y")
    np.save(target_fname, test_y)

    file = open(flag_file, "w")
    file.close()



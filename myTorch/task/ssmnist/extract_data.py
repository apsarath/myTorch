import numpy as np
import _pickle as pickle
import os

from myTorch.utils import create_folder


def create_targets(src_folder, tgt_folder):

    train = np.zeros(50000, dtype="float32")
    valid = np.zeros(10000, dtype="float32")
    test = np.zeros(10000, dtype="float32")

    file_name = os.path.join(src_folder, "trainlabels.txt")
    file = open(file_name, "r")
    for i in range(0, 50000):
        number = float(file.readline().strip())
        train[i] = number

    for i in range(0, 10000):
        number = float(file.readline().strip())
        valid[i] = number

    file.close()

    np.save(os.path.join(tgt_folder, "y_train"), train)
    np.save(os.path.join(tgt_folder, "y_valid"), valid)

    file_name = os.path.join(src_folder, "testlabels.txt")
    file = open(file_name, "r")
    for i in range(0, 10000):
        number = float(file.readline().strip())
        test[i] = number

    file.close()

    np.save(os.path.join(tgt_folder, "y_test"), test)


def create_sequence(src_folder, tgt_folder):

    train = list()
    valid = list()
    test = list()

    train_seq = np.zeros(50000, dtype="float32")
    valid_seq = np.zeros(10000, dtype="float32")
    test_seq = np.zeros(10000, dtype="float32")

    for i in range(0, 60000):
        print(i)
        file = open(src_folder + "trainimg-"+str(i)+"-inputdata.txt", "r")
        seq_list = list()
        start = True

        for line in file:
            line = line.strip().split()
            if start:
                a = 0
                b = 0
                c = float(line[2])
                d = float(line[3])
                start = False
            else:
                a = float(line[0])
                b = float(line[1])
                c = float(line[2])
                d = float(line[3])
            seq_list.append(np.asarray([a, b, c, d]))
        if i < 50000:
            train.append(seq_list)
            train_seq[i] = len(seq_list)
        else:
            valid.append(seq_list)
            valid_seq[i - 50000] = len(seq_list)
        file.close()

    pickle.dump(train, open(tgt_folder + "x_train.pkl", "wb"))
    pickle.dump(valid, open(tgt_folder + "x_valid.pkl", "wb"))
    np.save(tgt_folder + "seq_len_train", train_seq)
    np.save(tgt_folder + "seq_len_valid", valid_seq)
    print("train, valid done...")

    for i in range(0, 10000):
        print(i)
        file = open(src_folder + "testimg-"+str(i)+"-inputdata.txt", "r")
        seq_list = list()
        start = True

        for line in file:
            line = line.strip().split()
            if start:
                a = 0
                b = 0
                c = float(line[2])
                d = float(line[3])
                start = False
            else:
                a = float(line[0])
                b = float(line[1])
                c = float(line[2])
                d = float(line[3])
            seq_list.append(np.asarray([a, b, c, d]))
        test.append(seq_list)
        test_seq[i] = len(seq_list)
        file.close()

    pickle.dump(test, open(tgt_folder + "x_test.pkl", "wb"))
    np.save(tgt_folder + "seq_len_test", test_seq)


def create_multidigit_sequence(src_folder, tgt_folder, num_digits, num_train_data_points = 200000,
                               num_val_data_points=1000):

    np.random.seed(num_digits)

    x_train = pickle.load(open(src_folder+"x_train.pkl", "rb"))
    x_valid = pickle.load(open(src_folder+"x_valid.pkl", "rb"))
    x_test = pickle.load(open(src_folder+"x_test.pkl", "rb"))

    y_train = np.load(src_folder+"y_train.npy")
    y_valid = np.load(src_folder+"y_valid.npy")
    y_test = np.load(src_folder+"y_test.npy")

    create_folder(tgt_folder)

    train_seq = np.zeros(num_train_data_points, dtype="float32")
    valid_seq = np.zeros(num_val_data_points, dtype="float32")
    test_seq = np.zeros(num_val_data_points, dtype="float32")

    multi_x_train = list()
    multi_x_valid = list()
    multi_x_test = list()

    multi_y_train = np.zeros((num_train_data_points, num_digits + 1), dtype="float32")
    multi_y_valid = np.zeros((num_val_data_points, num_digits + 1), dtype="float32")
    multi_y_test = np.zeros((num_val_data_points, num_digits + 1), dtype="float32")

    for i in range(0, num_train_data_points):

        if i % 100 == 0:
            print(i)

        seq_list = list()
        for j in range(0, num_digits):

            digit_id = np.random.randint(0, 49999)
            seq_list.extend(x_train[digit_id])
            multi_y_train[i][j] = y_train[digit_id]

        train_seq[i] = len(seq_list)
        multi_y_train[i][-1] = 10
        multi_x_train.append(np.asarray(seq_list))

    np.save(tgt_folder + "seq_len_train", train_seq)
    np.save(tgt_folder + "y_train", multi_y_train)
    pickle.dump(multi_x_train, open(tgt_folder+"x_train.pkl", "wb"))

    for i in range(0, num_val_data_points):

        if i % 100 == 0:
            print(i)

        seq_list = list()
        for j in range(0, num_digits):

            digit_id = np.random.randint(0, 9999)
            seq_list.extend(x_valid[digit_id])
            multi_y_valid[i][j] = y_valid[digit_id]

        valid_seq[i] = len(seq_list)
        multi_y_valid[i][-1] = 10
        multi_x_valid.append(np.asarray(seq_list))

    np.save(tgt_folder + "seq_len_valid", valid_seq)
    np.save(tgt_folder + "y_valid", multi_y_valid)
    pickle.dump(multi_x_valid, open(tgt_folder+"x_valid.pkl", "wb"))

    for i in range(0, num_val_data_points):

        if i % 100 == 0:
            print(i)

        seq_list = list()
        for j in range(0, num_digits):

            digit_id = np.random.randint(0, 9999)
            seq_list.extend(x_test[digit_id])
            multi_y_test[i][j] = y_test[digit_id]

        test_seq[i] = len(seq_list)
        multi_y_test[i][-1] = 10
        multi_x_test.append(np.asarray(seq_list))

    np.save(tgt_folder + "seq_len_test", test_seq)
    np.save(tgt_folder + "y_test", multi_y_test)
    pickle.dump(multi_x_test, open(tgt_folder+"x_test.pkl", "wb"))


src_folder = "/mnt/data/sarath/data/ssmnist/sequences/"
tgt_folder = "/mnt/data/sarath/data/ssmnist/data/"


#create_targets(src_folder, tgt_folder)
#create_sequence(src_folder, tgt_folder)
# create_multidigit_sequence(tgt_folder, tgt_folder+"5/", 5)
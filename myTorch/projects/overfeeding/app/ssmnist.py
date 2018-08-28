import argparse

from myTorch.task.ssmnist.extract_data import create_multidigit_sequence
from myTorch.utils import create_config

parser = argparse.ArgumentParser(description="Algorithm Learning Task")
# parser.add_argument("--config", type=str, default="config/shagun/associative_recall.yaml", help="config file path.")
parser.add_argument("--config", type=str, default="../config/aaai/ssmnist/128.yaml", help="config file path.")
parser.add_argument("--force_restart", type=bool, default=False, help="if True start training from scratch.")
args = parser.parse_args()
# logging.basicConfig(level=logging.INFO, filename="log.txt", filemode="w")

src_folder = "/mnt/data/shagun/data/ssmnist/data/"
tgt_folder = None
config = create_config(args.config)
for num_digits in range(config.min_seq_len, config.max_seq_len, config.step_seq_len):
    tgt_folder = src_folder + str(num_digits)+"/"
    print(tgt_folder)
    create_multidigit_sequence(src_folder, tgt_folder, num_digits,
                               num_train_data_points=config.max_steps * config.batch_size,
                               num_val_data_points=config.evaluate_over_n * config.batch_size)
    print("Data created for num_digits = {}\n".format(num_digits))

# create_targets(src_folder, tgt_folder)
# create_sequence(src_folder, tgt_folder)
# create_multidigit_sequence(tgt_folder, tgt_folder+"5/", 5)

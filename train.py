"""Trains a chemprop model on a dataset."""
from chemprop.train import cross_validate, run_training
from chemprop.args import TrainArgs

# ---------------train-------------#

train_arguments = [
    '--data_path', f'/home/fuli/my_code/git/tox_data/tox_data_v2/multiple_task/all.csv',
    '--config_path', f'checkpoints/config/Basic.json',
    '--dataset_type', 'classification',
    '--save_dir', f'checkpoints/att/all_model',
    '--epochs', '60',
    '--batch_size', '256',
    '--num_workers', '0',
    '--gpu', '0',
]

args = TrainArgs().parse_args(train_arguments)
mean_score, std_score = cross_validate(args=args, train_func=run_training)

"""Trains a chemprop model on a dataset."""
import argparse
from chemprop.train import cross_validate, run_training
from chemprop.args import TrainArgs, HyperoptArgs
from chemprop.hyperparameter_optimization import hyperopt

task = 'Drug'

parser = argparse.ArgumentParser()
parser.add_argument('--file', help='file iter')
parser.add_argument('--gpu', help='gpu id')
input_args = parser.parse_args()

# --------------- Hyperparameter Optimization-------------#
hyperparameter_arguments = [
    '--data_path', f'/home/fuli/my_code/git/tox_data/tox_data_v1/{task}/cla/{input_args.file}/{task}_train.csv',
    '--separate_val_path', f'/home/fuli/my_code/git/tox_data/tox_data_v1/{task}/cla/{input_args.file}/{task}_valid.csv',
    '--separate_test_path', f'/home/fuli/my_code/git/tox_data/tox_data_v1/{task}/cla/{input_args.file}/{task}_test.csv',
    '--dataset_type', 'classification',
    '--hyperopt_checkpoint_dir', f'checkpoints/{task}/cla/{task}_hyperopt',
    '--gpu', str(input_args.gpu),
    '--batch_size', '128',
    '--num_iters', '3',
    '--epochs', '30',
    '--metric', 'auc',
    '--aggregation', 'norm',
    '--search_parameter_keywords', 'depth', 'ffn_num_layers', 'hidden_size', 'ffn_hidden_size', 'dropout',
    '--config_save_path', f'checkpoints/{task}/cla/{task}_hyperopt/config.json',
    '--log_dir', f'checkpoints/{task}/cla/{task}_hyperopt',
]
hy_args = HyperoptArgs().parse_args(hyperparameter_arguments)
hyperopt(args=hy_args)

# ---------------train-------------#
train_arguments = [
    '--data_path', f'/home/fuli/my_code/git/tox_data/tox_data_v1/{task}/cla/{input_args.file}/{task}_train.csv',
    '--separate_val_path', f'/home/fuli/my_code/git/tox_data/tox_data_v1/{task}/cla/{input_args.file}/{task}_valid.csv',
    '--separate_test_path', f'/home/fuli/my_code/git/tox_data/tox_data_v1/{task}/cla/{input_args.file}/{task}_test.csv',
    '--config_path', f'checkpoints/{task}/cla/{task}_hyperopt/config.json',
    '--dataset_type', 'classification',
    '--save_dir', f'checkpoints/{task}/cla/{task}_model',
    '--epochs', '200',
    '--gpu', str(input_args.gpu),
    '--save_smiles_splits',
    '--save_preds',
]

args = TrainArgs().parse_args(train_arguments)
mean_score, std_score = cross_validate(args=args, train_func=run_training)

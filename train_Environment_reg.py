"""Trains a chemprop model on a dataset."""
import argparse
from chemprop.train import cross_validate, run_training
from chemprop.args import TrainArgs, HyperoptArgs
from chemprop.hyperparameter_optimization import hyperopt

task = 'Environment_reg'
gpu = 1


# --------------- Hyperparameter Optimization-------------#
hyperparameter_arguments = [
    '--data_path', f'/home/fuli/my_code/git/tox_data/tox_data_v1/{task}/reg/{task}_train.csv',
    '--separate_val_path', f'/home/fuli/my_code/git/tox_data/tox_data_v1/{task}/reg/{task}_valid.csv',
    '--separate_test_path', f'/home/fuli/my_code/git/tox_data/tox_data_v1/{task}/reg/{task}_test.csv',
    '--dataset_type', 'regression',
    '--hyperopt_checkpoint_dir', f'checkpoints/{task}/reg/{task}_hyperopt',
    '--gpu', str(gpu),
    '--batch_size', '128',
    '--num_iters', '10',
    '--epochs', '300',
    '--aggregation', 'norm',
    '--search_parameter_keywords', 'depth', 'ffn_num_layers', 'hidden_size', 'ffn_hidden_size', 'dropout',
    '--config_save_path', f'checkpoints/{task}/reg/{task}_hyperopt/config.json',
    '--log_dir', f'checkpoints/{task}/reg/{task}_hyperopt',
]
hy_args = HyperoptArgs().parse_args(hyperparameter_arguments)
hyperopt(args=hy_args)

# ---------------train-------------#
parser = argparse.ArgumentParser()
parser.add_argument('--file', help='file iter')
parser.add_argument('--gpu', help='gpu id')
input_args = parser.parse_args()

train_arguments = [
    '--data_path', f'/home/fuli/my_code/git/tox_data/tox_data_v2/multiple_task/Environment_reg/{input_args.file}/train.csv',
    '--separate_val_path', f'/home/fuli/my_code/git/tox_data/tox_data_v2/multiple_task/Environment_reg/{input_args.file}/val.csv',
    '--separate_test_path', f'/home/fuli/my_code/git/tox_data/tox_data_v2/multiple_task/Environment_reg/{input_args.file}/test.csv',
    '--config_path', f'checkpoints/DMPNN/{task}/config.json',
    '--dataset_type', 'regression',
    # '--save_dir', f'checkpoints/DMPNN/{task}/{task}_{input_args.file}_model',
    '--save_dir', f'checkpoints/DMPNN/test_conform_reg_model',
    '--epochs', '1',
    '--gpu', str(input_args.gpu),
    '--loss_function', 'quantile_interval',
    '--show_individual_scores',
    '--save_preds',
]

args = TrainArgs().parse_args(train_arguments)
mean_score, std_score = cross_validate(args=args, train_func=run_training)

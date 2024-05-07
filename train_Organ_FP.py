"""Trains a chemprop model on a dataset."""
import argparse
from chemprop.train import cross_validate, run_training
from chemprop.args import TrainArgs, HyperoptArgs
from chemprop.hyperparameter_optimization import hyperopt

task = 'Organ'
gpu = 0


# --------------- Hyperparameter Optimization-------------#
# hyperparameter_arguments = [
#     '--data_path', f'/home/fuli/my_code/git/tox_data/tox_data_v1/{task}/{task}_train.csv',
#     '--separate_val_path', f'/home/fuli/my_code/git/tox_data/tox_data_v1/{task}/{task}_valid.csv',
#     '--separate_test_path', f'/home/fuli/my_code/git/tox_data/tox_data_v1/{task}/{task}_test.csv',
#     '--dataset_type', 'classification',
#     '--hyperopt_checkpoint_dir', f'checkpoints/{task}_hyperopt',
#     '--gpu', str(gpu),
#     '--batch_size', '128',
#     '--num_iters', '10',
#     '--epochs', '300',
#     '--metric', 'auc',
#     '--aggregation', 'norm',
#     '--search_parameter_keywords', 'depth', 'ffn_num_layers', 'hidden_size', 'ffn_hidden_size', 'dropout',
#     '--config_save_path', f'checkpoints/{task}_hyperopt/config.json',
#     '--log_dir', f'checkpoints/{task}_hyperopt',
# ]
# hy_args = HyperoptArgs().parse_args(hyperparameter_arguments)
# hyperopt(args=hy_args)

# ---------------train-------------#
parser = argparse.ArgumentParser()
parser.add_argument('--file', help='file iter')
parser.add_argument('--gpu', help='gpu id')
input_args = parser.parse_args()

train_arguments = [
    '--data_path', f'/home/fuli/my_code/git/tox_data/tox_data_v1/{task}/{input_args.file}/{task}_train.csv',
    '--separate_val_path', f'/home/fuli/my_code/git/tox_data/tox_data_v1/{task}/{input_args.file}/{task}_valid.csv',
    '--separate_test_path', f'/home/fuli/my_code/git/tox_data/tox_data_v1/{task}/{input_args.file}/{task}_test.csv',
    '--config_path', f'checkpoints/{task}/{task}_hyperopt/config.json',
    '--dataset_type', 'classification',
    '--save_dir', f'checkpoints/{task}/{task}_{input_args.file}_model',
    '--epochs', '1000',
    '--gpu', str(input_args.gpu),
    '--save_smiles_splits',
    '--save_preds',
]

args = TrainArgs().parse_args(train_arguments)
mean_score, std_score = cross_validate(args=args, train_func=run_training)

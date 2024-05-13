"""Trains a chemprop model on a dataset."""
import argparse
from chemprop.train import cross_validate, run_training
from chemprop.args import TrainArgs, HyperoptArgs
from chemprop.hyperparameter_optimization import hyperopt

task = 'Environments'
parser = argparse.ArgumentParser()
parser.add_argument('--file', help='file iter')
parser.add_argument('--gpu', help='gpu id')
input_args = parser.parse_args()


# --------------- Hyperparameter Optimization-------------#
# hyperparameter_arguments = [
#     '--data_path', f'/home/fuli/my_code/git/tox_data/tox_data_v1/{task}/reg/{input_args.file}/{task}_train.csv',
#     '--separate_val_path', f'/home/fuli/my_code/git/tox_data/tox_data_v1/{task}/reg/{input_args.file}/{task}_valid.csv',
#     '--separate_test_path', f'/home/fuli/my_code/git/tox_data/tox_data_v1/{task}/reg/{input_args.file}/{task}_test.csv',
#     '--features_path', f'/home/fuli/my_code/git/tox_data/tox_data_v1_FP/{task}/reg/{input_args.file}/{task}_train_2d.npy',
#     '--separate_val_features_path', f'/home/fuli/my_code/git/tox_data/tox_data_v1_FP/{task}/reg/{input_args.file}/{task}_valid_2d.npy',
#     '--separate_test_features_path', f'/home/fuli/my_code/git/tox_data/tox_data_v1_FP/{task}/reg/{input_args.file}/{task}_test_2d.npy',
#     '--dataset_type', 'regression',
#     '--hyperopt_checkpoint_dir', f'checkpoints/DMPNN_FP/{task}/reg/{task}_hyperopt',
#     '--gpu', str(input_args.gpu),
#     '--batch_size', '128',
#     '--num_iters', '10',
#     '--epochs', '300',
#     '--metric', 'auc',
#     '--aggregation', 'norm',
#     '--search_parameter_keywords', 'depth', 'ffn_num_layers', 'hidden_size', 'ffn_hidden_size', 'dropout',
#     '--config_save_path', f'checkpoints/DMPNN_FP/{task}/reg/{task}_hyperopt/config.json',
#     '--log_dir', f'checkpoints/DMPNN_FP/{task}/reg/{task}_hyperopt',
# ]
# hy_args = HyperoptArgs().parse_args(hyperparameter_arguments)
# hyperopt(args=hy_args)

# ---------------train-------------#

train_arguments = [
    '--data_path', f'/home/fuli/my_code/git/tox_data/tox_data_v1/{task}/reg/{input_args.file}/{task}_train.csv',
    '--separate_val_path', f'/home/fuli/my_code/git/tox_data/tox_data_v1/{task}/reg/{input_args.file}/{task}_valid.csv',
    '--separate_test_path', f'/home/fuli/my_code/git/tox_data/tox_data_v1/{task}/reg/{input_args.file}/{task}_test.csv',
    '--features_path', f'/home/fuli/my_code/git/tox_data/tox_data_v1_FP/{task}/reg/{input_args.file}/{task}_train_2d.npy',
    '--separate_val_features_path', f'/home/fuli/my_code/git/tox_data/tox_data_v1_FP/{task}/reg/{input_args.file}/{task}_valid_2d.npy',
    '--separate_test_features_path', f'/home/fuli/my_code/git/tox_data/tox_data_v1_FP/{task}/reg/{input_args.file}/{task}_test_2d.npy',
    '--config_path', f'checkpoints/DMPNN/{task}/reg/{task}_hyperopt/config.json',
    '--dataset_type', 'regression',
    '--save_dir', f'checkpoints/DMPNN_FP/{task}/reg/{task}_{input_args.file}_model',
    '--epochs', '300',
    '--gpu', str(input_args.gpu),
    '--save_smiles_splits',
    '--save_preds',
]

args = TrainArgs().parse_args(train_arguments)
mean_score, std_score = cross_validate(args=args, train_func=run_training)

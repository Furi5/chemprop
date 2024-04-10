"""Trains a chemprop model on a dataset."""
import chemprop
from chemprop.train import chemprop_train


if __name__ == '__main__':
    arguments = [
    '--data_path', 'data/clintox.csv',
    '--dataset_type', 'classification',
    '--save_dir', 'test_checkpoints_cla',
    '--epochs', '5',
    '--save_smiles_splits',
    '--no_cuda',
]

    args = chemprop.args.TrainArgs().parse_args(arguments)
    mean_score, std_score = chemprop.train.cross_validate(args=args, train_func=chemprop.train.run_training)

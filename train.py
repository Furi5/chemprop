"""Trains a chemprop model on a dataset."""
import chemprop
from chemprop.train import chemprop_train


if __name__ == '__main__':
    arguments = [
        '--data_path', 'data/clintox.csv',
        '--dataset_type', 'classification',
        '--save_dir', 'test_checkpoints_cla',
        '--epochs', '10',
        '--save_smiles_splits',
        '--no_cuda',
    ]

    args = chemprop.args.TrainArgs().parse_args(arguments)
    mean_score, std_score = chemprop.train.cross_validate(
        args=args, train_func=chemprop.train.run_training)

    # arguments = [
    #     '--test_path', '/dev/null',
    #     '--preds_path', '/dev/null',
    #     '--checkpoint_dir', 'test_checkpoints_cla'
    # ]

    # args = chemprop.args.PredictArgs().parse_args(arguments)

    # model_objects = chemprop.train.load_model(args=args)

    # smiles = [['CCC'], ['CCCC'], ['OCC']]
    # preds = chemprop.train.make_predictions(args=args, smiles=smiles, model_objects=model_objects)
    # print(preds)

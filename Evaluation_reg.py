import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


def regressor_metrics(y_test, y_pred):
    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    regression_metrics = {
        'r2': r2,
        'rmse': rmse,
        'mae': mae,
    }
    return regression_metrics


def multi_task_evaluation(test_data, predict_data):
    task_name = test_data.columns[1:]
    df = pd.DataFrame()
    test_data.reset_index(drop=True, inplace=True)
    predict_data.reset_index(drop=True, inplace=True)
    for task in task_name:
        value_index = test_data[test_data[task].notna()].index
        y_true = test_data.loc[value_index, task]
        y_pred = predict_data.loc[value_index, task]
        task_metrics = regressor_metrics(y_true, y_pred)
        task_metrics['task'] = task
        task_metrics = pd.DataFrame(task_metrics, index=[0])
        df = pd.concat([df, task_metrics], axis=0)
    return df


def group_df(df_cla):
    grouped_cla_mean = df_cla.groupby('task').mean()
    grouped_cla_std = df_cla.groupby('task').std()
    grouped_cla_mean.reset_index(inplace=True)
    grouped_cla_std.reset_index(inplace=True)
    grouped_cla_mean = grouped_cla_mean.round(3)
    grouped_cla_std = grouped_cla_std.round(3)
    cla_columns = grouped_cla_mean.columns
    cla_mean_std = {}
    for col in cla_columns:
        cla_mean_std[col] = grouped_cla_mean[col].astype(
            str) + '±' + grouped_cla_std[col].astype(str)
    cla_mean_std['task'] = grouped_cla_mean['task']
    cla_mean_std = pd.DataFrame(cla_mean_std)
    cla_mean_std = cla_mean_std.set_index('task')
    cla_mean_std = cla_mean_std.reindex(columns=[
        'r2', 'rmse', 'mae'
    ])
    return cla_mean_std


def random_multiple(task, test_path, pred_path, split_times, output_path):
    df = pd.DataFrame()
    for i in range(split_times):
        i += 1
        test_file = f'{test_path}/{i}/test.csv'
        pred_file = f'{pred_path}/{task}_{i}_model/test_preds.csv'
        test_data = pd.read_csv(test_file)
        predict_data = pd.read_csv(pred_file)
        test_df = multi_task_evaluation(test_data, predict_data)
        df = pd.concat([df, test_df], axis=0)
        print(i, 'done')
    df.to_csv(f'{output_path}_10.csv', index=False)
    mean_sd = group_df(df)
    mean_sd.to_csv(f'{output_path}_10_mean_sd.csv')


if __name__ == '__main__':
    random_multiple(
        task='Environment_reg',
        test_path='/home/fuli/my_code/git/tox_data/tox_data_v2/multiple_task/Environment_reg',
        pred_path='/home/fuli/my_code/git/chemprop/checkpoints/DMPNN_Des/Environment_reg',
        split_times=10,
        output_path='ModelPerformance/DMPNN_Des/reg/Environment_reg')

    random_multiple(
        task='Basic_reg',
        test_path='/home/fuli/my_code/git/tox_data/tox_data_v2/multiple_task/Basic_reg',
        pred_path='/home/fuli/my_code/git/chemprop/checkpoints/DMPNN_Des/Basic_reg',
        split_times=10,
        output_path='ModelPerformance/DMPNN_Des/reg/Basic_reg')

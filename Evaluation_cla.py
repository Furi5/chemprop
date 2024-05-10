import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import precision_recall_curve, f1_score

drug_order = [
    "Carcinogenicity",
    "developmental_toxicity",
    "FDAMDD",
    "Genotoxicity",
    "Hemolytic",
    "Mitochondrial",
    "neurotoxicity",
    "ROA",
    "RPMI_8226",
    "reproductive",
    "Respiratory",
    "TA100",
    "TA100_S9",
    "TA102",
    "TA102_S9",
    "TA104",
    "TA104_S9",
    "TA137_S9",
    "TA1535",
    "TA1535_S9",
    "TA1537",
    "TA1538",
    "TA1538_S9",
    "TA97",
    "TA97_S9",
    "TA98",
    "TA98_S9",
]


def find_best_threshold(y_true, y_pred_probs):
    precision, recall, thresholds = precision_recall_curve(
        y_true, y_pred_probs)
    # 避免除以零，将分母设为一个小的值
    epsilon = 1e-10
    f1_scores = 2 * (precision * recall) / (precision + recall + epsilon)
    # 找到最大 F1 分数对应的阈值
    best_threshold = thresholds[f1_scores.argmax()]
    return best_threshold


def ClassMetrics(y_true, y_pred_s):
    # 模型的苹评估
    y_pred = []
    y_test = []
    y_pred_p = []
    # threshold = find_best_threshold(y_true, y_pred_s)
    # print('threshold:', threshold)
    threshold = 0.5
    for s, t in zip(y_pred_s, y_true):
        try:
            s = float(s)
            if s >= threshold:
                y_pred.append(1)
            else:
                y_pred.append(0)
            y_test.append(t)
            y_pred_p.append(s)
        except ValueError:
            print("无法将字符串转换为浮点数：无效的输入")
    acc = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_p)
    TN, FP, FN, TP = confusion_matrix(y_test, y_pred).ravel()
    SE = TP / (TP + FN)  # 敏感度是实际为正类的样本中，被模型正确预测为正类的比例
    SP = TN / (TN + FP)  # 特异性是实际为负类的样本中，被模型正确预测为负类的比例。
    MCC = matthews_corrcoef(y_test, y_pred)
    f1_label1 = f1_score(y_test, y_pred, pos_label=1)
    precision_label1 = precision_score(y_test, y_pred, pos_label=1)
    f1_label0 = f1_score(y_test, y_pred, pos_label=0)
    precision_label0 = precision_score(y_test, y_pred, pos_label=0)
    balanced_accuracy = balanced_accuracy_score(y_test, y_pred)
    classification_metrics = {
        'balanced_accuracy': round(balanced_accuracy, 4),
        'AUC': round(roc_auc, 4),
        'ACC': round(acc, 4),
        'SP': round(SP, 4),
        'SE': round(SE, 4),
        'MCC': round(MCC, 4),
        'f1_label1': round(f1_label1, 4),
        'f1_label0': round(f1_label0, 4),
        'precision_label1': round(precision_label1, 4),
        'precision_label0': round(precision_label0, 4)
    }
    return classification_metrics


def multi_task_evaluation(test_data, predict_data):
    task_name = test_data.columns[1:]
    df = pd.DataFrame()
    test_data.reset_index(drop=True, inplace=True)
    predict_data.reset_index(drop=True, inplace=True)
    for task in task_name:
        value_index = test_data[test_data[task].notna()].index
        y_true = test_data.loc[value_index, task]
        y_pred = predict_data.loc[value_index, task]
        task_metrics = ClassMetrics(y_true, y_pred)
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
        "AUC", "ACC", "SP", "SE", "MCC", "f1_label1", "f1_label0", "precision_label1", "precision_label0"
    ])
    return cla_mean_std


def random_multiple(task, test_path, pred_path, split_times, output_path):
    df = pd.DataFrame()
    for i in range(split_times):
        i += 1
        if task in ['Cell', 'Drug']:
            test_file = f'{test_path}/{i}/test.csv'
        else:
            test_file = f'{test_path}/{i}/{task}_test.csv'
        pred_file = f'{pred_path}/{task}_{i}_model/test_preds.csv'
        test_data = pd.read_csv(test_file)
        predict_data = pd.read_csv(pred_file)
        test_df = multi_task_evaluation(test_data, predict_data)
        df = pd.concat([df, test_df], axis=0)
        print(i, 'done')
    if task == 'Drug':
        df['task'] = pd.Categorical(
            df['task'], categories=drug_order, ordered=True)

    mean_sd = group_df(df)
    mean_sd.to_csv(f'{output_path}_10_mean_sd.csv')
    df.to_csv(f'{output_path}_10.csv', index=False)


if __name__ == '__main__':
    random_multiple(
        task='Target',
        test_path='/home/fuli/my_code/git/tox_data/tox_data_v1/Target',
        pred_path='/home/fuli/my_code/git/chemprop/checkpoints/att/Target',
        split_times=1,
        output_path='ModelPerformance/att/cla/Target')

    random_multiple(
        task='Clinical',
        test_path='/home/fuli/my_code/git/tox_data/tox_data_v1/Clinical',
        pred_path='/home/fuli/my_code/git/chemprop/checkpoints/att/Clinical',
        split_times=1,
        output_path='ModelPerformance/att/cla/Clinical')

    random_multiple(
        task='Environments',
        test_path='/home/fuli/my_code/git/tox_data/tox_data_v1/Environments/cla',
        pred_path='/home/fuli/my_code/git/chemprop/checkpoints/att/Environments/cla',
        split_times=1,
        output_path='ModelPerformance/att/cla/Environments')

    random_multiple(
        task='Organ',
        test_path='/home/fuli/my_code/git/tox_data/tox_data_v1/Organ',
        pred_path='/home/fuli/my_code/git/chemprop/checkpoints/att/Organ',
        split_times=1,
        output_path='ModelPerformance/att/cla/Organ')

    random_multiple(
        task='Pathway',
        test_path='/home/fuli/my_code/git/tox_data/tox_data_v1/Pathway',
        pred_path='/home/fuli/my_code/git/chemprop/checkpoints/att/Pathway',
        split_times=1,
        output_path='ModelPerformance/att/cla/Pathway')

    random_multiple(
        task='Cell',
        test_path='/home/fuli/my_code/git/tox_data/tox_data_v2/multiple_task/Cell',
        pred_path='/home/fuli/my_code/git/chemprop/checkpoints/att/Cell',
        split_times=1,
        output_path='ModelPerformance/att/cla/Cell')

    random_multiple(
        task='Drug',
        test_path='/home/fuli/my_code/git/tox_data/tox_data_v2/multiple_task/Basic',
        pred_path='/home/fuli/my_code/git/chemprop/checkpoints/att/Drug/cla',
        split_times=1,
        output_path='ModelPerformance/att/cla/Drug')

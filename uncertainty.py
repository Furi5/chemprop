
import pandas as pd
from sklearn.metrics import confusion_matrix
import json


def cal_Youden_index(y_true, y_pred):
    threshold = 0.5  # Define a threshold for binarization

    # Binarize continuous predictions based on the threshold
    # y_pred = (y_pred >= threshold).astype(int)
    # 计算混淆矩阵
    conf_matrix = confusion_matrix(y_true, y_pred)

    # 提取混淆矩阵的 True Positive（TP）、True Negative（TN）、False Positive（FP）、False Negative（FN）
    TP = conf_matrix[1, 1]
    TN = conf_matrix[0, 0]
    FP = conf_matrix[0, 1]
    FN = conf_matrix[1, 0]

    # 计算敏感性（True Positive Rate）和特异性（True Negative Rate）
    sensitivity = TP / (TP + FN)
    specificity = TN / (TN + FP)

    # 计算Youden指数
    Youden_index = sensitivity + specificity - 1
    return Youden_index


data_dict = {
    'task': [],
    'max_yuden_index': [],
    'uncar_threshold': []
}


colnames_dict = {
    'Organ': [
        "Nephrotoxicity",
        "Hematotoxicity",
        "Eye_Irritation",
        "Eye_Corrosion",
        "Arhhythmia",
        "hepatotoxicity",
        "Hypertension",
        "DILI",
        "Myocardial",
        "Skinsen",
        "Ototoxicity",
        "Cardiacfailure",
        "hERG_II",
        "Nav15",
        "Cav12",
        "hERG_III",
        "Rhabdomyolysis",
        "hERG_I",
        "H-HT"
    ],
    'Basic': [
        "Carcinogenicity",
        "developmental_toxicity",
        "FDAMDD",
        "Genotoxicity",
        "Hemolytic_toxicity",
        "Mitochondrial_toxicity",
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
        "TA98_S9"
    ],
    'Cell': [
        "A549",
        "BJeLR",
        "Ba_Slash_F3",
        "H1299",
        "H69AR",
        "HEK293",
        "HEPG2",
        "HMLE_sh_Ecad",
        "HPDE-C7",
        "HT1080",
        "HUVEC",
        "IEC_6",
        "Jurkat",
        "KKLEB",
        "LL47",
        "SK-BR-3"
    ],
    'Clinical': [
        "FDA_APPROVED",
        "CT_TOX",
        "Hepatobiliary_disorders",
        "Metabolism_and_nutrition_disorders",
        "Product_issues",
        "Eye_disorders",
        "Investigations",
        "Musculoskeletal_and_connective_tissue_disorders",
        "Gastrointestinal_disorders",
        "Social_circumstances",
        "Immune_system_disorders",
        "Reproductive_system_and_breast_disorders",
        "Neoplasms_benign_malignant_and_unspecified_(incl_cysts_and_polyps)",
        "General_disorders_and_administration_site_conditions",
        "Endocrine_disorders",
        "Surgical_and_medical_procedures",
        "Vascular_disorders",
        "Blood_and_lymphatic_system_disorders",
        "Skin_and_subcutaneous_tissue_disorders",
        "Congenital_familial_and_genetic_disorders",
        "Infections_and_infestations",
        "Respiratory_thoracic_and_mediastinal_disorders",
        "Psychiatric_disorders",
        "Renal_and_urinary_disorders",
        "Pregnancy_puerperium_and_perinatal_conditions",
        "Ear_and_labyrinth_disorders",
        "Cardiac_disorders",
        "Nervous_system_disorders",
        "Injury_poisoning_and_procedural_complications"
    ],
    'Environments': [
        "Crustacean_EC50",
        # "Honey_bee_toxicity",
        # "LC50_Mallard_Duck",
        "TPT",
        "Biodegradability"
    ],
    'Pathway': [
        "NR-AR-LBD",
        "SR-p53",
        "NR-ER-LBD",
        "NR-Aromatase",
        "SR-ARE",
        "NR-AR",
        "SR-HSE",
        "SR-MMP",
        "SR-ATAD5",
        "NR-ER",
        "NR-PPAR-gamma",
        "NR-AhR"
    ],
    'Target': [
        "ADRB3",
        "ADRB2",
        "CHRM1",
        "AR",
        "ADRA1A",
        "MAOA",
        "CHRM2",
        "HTR1A",
        "DRD2",
        "HTR2A",
        "CHRM3",
        "ADORA3",
        "CCKAR",
        "HRH2",
        "EDNRB",
        "ADRA2A",
        "EDNRA",
        "HTR2B",
        "OPRD",
        "OPRM",
        "ADORA1",
        "HTR3A",
        "GABRA1",
        "BDKRB2",
        "NPY1R",
        "OPRK",
        "DRD1",
        "HTR2C",
        "PDE3",
        "ADRB1",
        "AGTR1",
        "ADRA2B",
        "HRH1",
        "ADORA2A"
    ]
}

data_path = '/home/fuli/my_code/git/chemprop/Calibration_set'


data_dict = {
    'task': [],
    'max_yuden_index': [],
    'uncar_threshold': []
}

for tasks in colnames_dict.keys():
    print(f'Processing {tasks}...')
    test_set = pd.read_csv(
        f'/home/fuli/my_code/git/tox_data/Calibration_test_set/{tasks}.csv')
    pred_set = pd.read_csv(
        f'{data_path}/{tasks}.csv')
    for task in colnames_dict[tasks]:
        print(f'Processing {task}...')
        y_true = test_set[task].tolist()
        y_pred = pred_set[task].tolist()
        y_pred = [1 if x >= 0.5 else 0 for x in y_pred]
        uncal = pred_set[f'{task}_uncertainty'].tolist()
        sorted_df = pd.concat(
            [pd.Series(y_pred, name='y_pred'), pd.Series(uncal, name='uncla'), pd.Series(y_true, name='y_true')], axis=1)

        sorted_df = sorted_df.sort_values(by='uncla', ascending=True)
        sorted_df = sorted_df.dropna(subset=['y_true'])
        print(sorted_df.head())

        Youden_index_list = []
        Correct_type = []
        for i in range(len(sorted_df)):
            if sorted_df.iloc[i, :]['y_true'] == sorted_df.iloc[i, :]['y_pred']:
                Correct_type.append(1)
            else:
                Correct_type.append(0)

        sorted_df['Correct_type'] = Correct_type
        auc = []
        for i in range(len(sorted_df)):
            # confidence
            sorted_df['confidence'] = (
                sorted_df['uncla'] <= sorted_df.iloc[i, :]['uncla']).astype(int)
            # 正确率
            Youden_index = cal_Youden_index(
                sorted_df['Correct_type'], sorted_df['confidence'])
            Youden_index_list.append(Youden_index)
            if Youden_index == max(Youden_index_list):
                uncar_threshold = sorted_df.iloc[i, :]['uncla']

        data_dict[task] = uncar_threshold
        # data_dict['max_yuden_index'].append(max(Youden_index_list))
        # data_dict['uncar_threshold'].append(uncar_threshold)

output = pd.DataFrame(data_dict)
# output.to_csv('uncertainty_threshold.csv', index=False)

with open('uncertainty_threshold.json', 'w') as json_file:
    json.dump(data_dict, json_file, indent=4)  # indent=4 使 JSON 文件更具可读性

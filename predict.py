"""Loads a trained chemprop model checkpoint and makes predictions on a dataset."""
import torch
import os
import pandas as pd
import chemprop
from Process_runner import ProcessRunner


uncertainty_threshold = {
    "Nephrotoxicity": 0.0034231929573769,
    "Hematotoxicity": 0.0124544579255805,
    "Eye_Irritation": 0.002323137682984,
    "Eye_Corrosion": 0.0616351211416292,
    "Arhhythmia": 0.0023066396036874,
    "hepatotoxicity": 0.006816030299375,
    "Hypertension": 0.0036408697934923,
    "DILI": 0.0029462002876891,
    "Myocardial": 0.0061348142751812,
    "Skinsen": 0.0038153577378539,
    "Ototoxicity": 0.0081906389482942,
    "Cardiacfailure": 0.0003242461653717,
    "hERG_II": 0.0047308076895232,
    "Nav15": 0.0037873732490292,
    "Cav12": 0.0038338944351348,
    "hERG_III": 0.0052990052323453,
    "Rhabdomyolysis": 7.278027223428936e-05,
    "hERG_I": 0.0023790680718255,
    "H-HT": 0.0019312112591656,
    "Carcinogenicity": 0.0222283640101472,
    "developmental_toxicity": 0.0078108454633739,
    "FDAMDD": 0.02394348228893,
    "Genotoxicity": 0.049598499096733,
    "Hemolytic_toxicity": 0.0176992175265958,
    "Mitochondrial_toxicity": 0.003853878929995,
    "neurotoxicity": 0.0159628034554351,
    "ROA": 0.0148200232178142,
    "RPMI_8226": 0.0009254225470554,
    "reproductive": 0.0236417178529518,
    "Respiratory": 0.0287155565122092,
    "TA100": 0.0008722664981166,
    "TA100_S9": 0.006676217379618,
    "TA102": 0.0072842004125716,
    "TA102_S9": 1.139390043311402e-05,
    "TA104": 0.0281657800191494,
    "TA104_S9": 0.0002983827294572,
    "TA137_S9": 0.000513354748338,
    "TA1535": 5.285309936623744e-06,
    "TA1535_S9": 0.0001537075318173,
    "TA1537": 2.154181720603887e-05,
    "TA1538": 0.0003763364420023,
    "TA1538_S9": 0.0025947394874954,
    "TA97": 0.0730959998680845,
    "TA97_S9": 0.0044095912246164,
    "TA98": 0.0026148835817166,
    "TA98_S9": 0.0021556830090634,
    "A549": 0.0005271588265361,
    "BJeLR": 0.0192769014017111,
    "Ba_Slash_F3": 0.0075381459739345,
    "H1299": 0.0027597783364054,
    "H69AR": 0.0024587258083342,
    "HEK293": 0.0021551970289928,
    "HEPG2": 0.0083879998915387,
    "HMLE_sh_Ecad": 0.0010369917416602,
    "HPDE-C7": 0.0013748186723112,
    "HT1080": 0.0002374750492775,
    "HUVEC": 0.0020856822577417,
    "IEC_6": 0.0051207444230848,
    "Jurkat": 0.0049487516442516,
    "KKLEB": 0.0020139794085211,
    "LL47": 0.0078128685152702,
    "SK-BR-3": 0.014576891488608,
    "FDA_APPROVED": 6.219334279800037e-07,
    "CT_TOX": 3.2016043592945276e-09,
    "Hepatobiliary_disorders": 0.002978821964783,
    "Metabolism_and_nutrition_disorders": 0.0006001030110077,
    "Product_issues": 6.651964054618043e-06,
    "Eye_disorders": 0.0050722005134076,
    "Investigations": 0.0023557966218994,
    "Musculoskeletal_and_connective_tissue_disorders": 0.0056468393425475,
    "Gastrointestinal_disorders": 6.015432366701656e-05,
    "Social_circumstances": 8.527583468565263e-07,
    "Immune_system_disorders": 0.0016780603313801,
    "Reproductive_system_and_breast_disorders": 0.0010665563926225,
    "Neoplasms_benign_malignant_and_unspecified_(incl_cysts_and_polyps)": 0.0072276671341594,
    "General_disorders_and_administration_site_conditions": 0.0034998377338781,
    "Endocrine_disorders": 8.011995063628152e-05,
    "Surgical_and_medical_procedures": 1.806351140523281e-05,
    "Vascular_disorders": 0.0029970836220484,
    "Blood_and_lymphatic_system_disorders": 0.0022469517552335,
    "Skin_and_subcutaneous_tissue_disorders": 0.0005539356580609,
    "Congenital_familial_and_genetic_disorders": 0.0284352932786124,
    "Infections_and_infestations": 0.0048647847037242,
    "Respiratory_thoracic_and_mediastinal_disorders": 0.0082663046513475,
    "Psychiatric_disorders": 0.0073433371444617,
    "Renal_and_urinary_disorders": 0.0085412117888389,
    "Pregnancy_puerperium_and_perinatal_conditions": 4.351694843481005e-07,
    "Ear_and_labyrinth_disorders": 0.002953428653975,
    "Cardiac_disorders": 0.0048908875405672,
    "Nervous_system_disorders": 8.777236512325093e-05,
    "Injury_poisoning_and_procedural_complications": 0.0017451197186651,
    "Crustacean_EC50": 0.0012137749162328,
    "TPT": 0.0002966474297387,
    "Biodegradability": 0.0001741956551593,
    "NR-AR-LBD": 0.0411190831689781,
    "SR-p53": 0.0010714186740969,
    "NR-ER-LBD": 0.0091956521577091,
    "NR-Aromatase": 0.0003830195745217,
    "SR-ARE": 0.0122150809307045,
    "NR-AR": 0.0151749321924625,
    "SR-HSE": 0.04148589022,
    "SR-MMP": 0.011905666294275,
    "SR-ATAD5": 7.612649047145907e-05,
    "NR-ER": 0.0237733419631429,
    "NR-PPAR-gamma": 0.0014094893272398,
    "NR-AhR": 0.0060549193454551,
    "ADRB3": 1.6533144969189735e-05,
    "ADRB2": 0.0012157231885905,
    "CHRM1": 0.0014373242838106,
    "AR": 0.009579487601005,
    "ADRA1A": 0.0047557064466926,
    "MAOA": 0.0025466566676586,
    "CHRM2": 0.0007374507702958,
    "HTR1A": 0.000778053557401,
    "DRD2": 0.0016265004098489,
    "HTR2A": 0.0002505354110157,
    "CHRM3": 0.0062281898881615,
    "ADORA3": 0.0001522306428828,
    "CCKAR": 0.0001728154841353,
    "HRH2": 0.0117006487237224,
    "EDNRB": 0.0005029150550128,
    "ADRA2A": 0.0024594819192183,
    "EDNRA": 0.0040320063617159,
    "HTR2B": 0.0010583467543409,
    "OPRD": 0.0002817879052416,
    "OPRM": 0.0002412913675635,
    "ADORA1": 0.000162062087869,
    "HTR3A": 0.000936832246894,
    "GABRA1": 0.0092509480408981,
    "BDKRB2": 0.000787819847523,
    "NPY1R": 0.006818327612462,
    "OPRK": 0.0003515576138914,
    "DRD1": 0.0017818604976889,
    "HTR2C": 0.0001744549019464,
    "PDE3": 0.0034012493424041,
    "ADRB1": 1.8400323572380715e-05,
    "AGTR1": 0.0001089074108087,
    "ADRA2B": 0.0160203933525775,
    "HRH1": 0.0009702761362714,
    "ADORA2A": 0.0010931189819086,
    "Honey_bee_toxicity": 0,
    "LC50_Mallard_Duck": 0,
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
    'Basic_reg': [
        "LOAEL",
        "ROA_LD50",
        "MRTD"
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
        "Honey_bee_toxicity",
        "LC50_Mallard_Duck",
        "TPT",
        "Biodegradability"
    ],
    'Environment_reg': [
        "BCF",
        "algae_pEC50",
        "crustaceans_pLC50",
        "fish_pLC50",
        "IBC50",
        "LC50DM",
        "LC50FM"
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


class suppress_stdout_stderr(object):
    '''
    A context manager for doing a "deep suppression" of stdout and stderr in
    Python, i.e. will suppress all print, even if the print originates in a
    compiled C/Fortran sub-function.
       This will not suppress raised exceptions, since exceptions are printed
    to stderr just before a script exits, and after the context manager has
    exited (at least, I think that is why it lets exceptions through).

    '''

    def __init__(self):
        # Open a pair of null files
        self.null_fds = [os.open(os.devnull, os.O_RDWR) for x in range(2)]
        # Save the actual stdout (1) and stderr (2) file descriptors.
        self.save_fds = (os.dup(1), os.dup(2))

    def __enter__(self):
        # Assign the null pointers to stdout and stderr.
        os.dup2(self.null_fds[0], 1)
        os.dup2(self.null_fds[1], 2)

    def __exit__(self, *_):
        # Re-assign the real stdout/stderr back to (1) and (2)
        os.dup2(self.save_fds[0], 1)
        os.dup2(self.save_fds[1], 2)
        # Close the null files
        os.close(self.null_fds[0])
        os.close(self.null_fds[1])


def classify_uncertainty(value, uncal_threshold):
    if value == 'Invalid SMILES':
        classified_uncertainty = "Invalid SMILES"
    elif value > uncal_threshold:
        classified_uncertainty = "Low-confidence"
    else:
        classified_uncertainty = "High-confidence"
    return classified_uncertainty


def tox_predict(task,
                smiles_file,
                pred_file,
                smiles_list):
    arguments = [
        '--test_path', smiles_file,
        '--preds_path', pred_file,
        '--checkpoint_paths', f'/home/fuli/my_code/git/chemprop/checkpoints/DMPNN/{task}/{task}_2_model/fold_0/model_0/model.pt',
        '--num_workers', '0',
        '--uncertainty_method', 'dropout',
        # '--no_cuda'
    ]

    args = chemprop.args.PredictArgs().parse_args(arguments)
    preds, un = chemprop.train.make_predictions(
        args=args, smiles=[[smi] for smi in smiles_list], return_uncertainty=True)

    sub_df = pd.DataFrame(
        preds,
        columns=colnames_dict[task],
        index=smiles_list
    )

    # if task not in ['Environment_reg', 'Basic_reg']:
    #     uncal = []
    #     for un_mol in un:
    #         un_mols = []
    #         for t, uncertainty in zip(colnames_dict[task], un_mol):
    #             un_mols.append(classify_uncertainty(
    #                 uncertainty, uncertainty_threshold[t]))
    #         uncal.append(un_mols)
    # else:
    #     uncal = un

    sub_uncertainty = pd.DataFrame(
        un,
        columns=[col+'_uncertainty' for col in colnames_dict[task]],
        index=smiles_list
    )

    cols = [col for cols in zip(
        colnames_dict[task], [col+'_uncertainty' for col in colnames_dict[task]]) for col in cols]

    sub_df = pd.concat([sub_df, sub_uncertainty], axis=1)
    sub_df = sub_df[cols]

    return sub_df


def main(smiles_list):
    with ProcessRunner() as P:
        pred_file = 'output.csv'
        smiles_file = 'input.csv'
        smi_df = pd.DataFrame({'smiles': smiles_list})
        smi_df.to_csv(smiles_file, index=False)
        all_preds = pd.DataFrame()
        for task in colnames_dict.keys():
            preds_df = tox_predict(task,
                                   smiles_file,
                                   pred_file,
                                   smiles_list
                                   )
            all_preds = pd.concat([all_preds, preds_df], axis=1)
            all_preds.index.name = 'smiles'
        rows_to_remove = ["Honey_bee_toxicity", "Honey_bee_toxicity_uncertainty",
                          "LC50_Mallard_Duck", "LC50_Mallard_Duck_uncertainty"]
        all_preds = all_preds.drop(rows_to_remove, axis=1)

    return all_preds


if __name__ == '__main__':
    import time
    import pandas as pd

    # smiles_list = ["CC(C)OC(=O)CC(=O)CSc1nc2c(cc1C#N)CCC2", "123"]*1000

    # start = time.time()
    # with suppress_stdout_stderr():  # suppress_stdout_stderr() 用于屏蔽 chemprop 的输出
    #     preds_df = main(smiles_list)

    # end = time.time()
    # print('Time:', end-start)
    # print(preds_df)
    for task in colnames_dict.keys():
        df = pd.read_csv(
            f'/home/fuli/my_code/git/tox_data/Calibration_test_set/{task}.csv')
        smiles_list = df['smiles'].tolist()
        preds_df = main(smiles_list)
        preds_df.to_csv(
            f'/home/fuli/my_code/git/chemprop/Calibration_set/{task}.csv', index=False)

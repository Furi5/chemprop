"""Loads a trained chemprop model checkpoint and makes predictions on a dataset."""
import torch
import os
import pandas as pd
import chemprop
from chemprop.nn_utils import visualize_atom_attention, attention_tensor_np
from chemprop.Process_runner import ProcessRunner


colnames_dict = {
    'organ': [
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
    'drug_cla': [
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
    'drug_reg': [
        "LOAEL",
        "ROA_LD50",
        "MRTD"
    ],
    'cell': [
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
    'clinical': [
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
    'environment_cla': [
        "Crustacean_EC50",
        "Honey_bee_toxicity",
        "LC50_Mallard_Duck",
        "TPT",
        "Biodegradability"
    ],
    'environment_reg': [
        "BCF",
        "algae_pEC50",
        "crustaceans_pLC50",
        "fish_pLC50",
        "IBC50",
        "LC50DM",
        "LC50FM"
    ],
    'pathway': [
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
    'target': [
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


def tox_predict(task,
                smiles_file,
                pred_file,
                smiles_list):
    arguments = [
        '--test_path', smiles_file,
        '--preds_path', pred_file,
        '--checkpoint_paths', f'/home/websites/deepToxLab/deepToxLab-backend/api/chemprop_att/checkpoints_att/{task}.pt',
        # '--checkpoint_paths', f'/home/fuli/my_code/git/chemprop/checkpoints_att/{task}.pt',
        "--num_workers", "0",
        '--uncertainty_method', 'dropout',
        "--no_cuda"
    ]

    args = chemprop.args.PredictArgs().parse_args(arguments)
    preds, un, att = chemprop.train.make_predictions(
        args=args, smiles=[[smi] for smi in smiles_list], return_uncertainty=True)

    sub_df = pd.DataFrame(
        preds,
        columns=colnames_dict[task],
        index=smiles_list
    )

    sub_uncertainty = pd.DataFrame(
        un,
        columns=[col+'_uncertainty' for col in colnames_dict[task]],
        index=smiles_list
    )

    cols = [col for cols in zip(
        colnames_dict[task], [col+'_uncertainty' for col in colnames_dict[task]]) for col in cols]

    sub_df = pd.concat([sub_df, sub_uncertainty], axis=1)
    sub_df = sub_df[cols]

    attention_df = pd.DataFrame(
        att,
        columns=colnames_dict[task],
        index=smiles_list
    )

    return sub_df, attention_df


def main(smiles_list):
    with ProcessRunner() as P:
        pred_file = 'output.csv'
        smiles_file = 'input.csv'
        smi_df = pd.DataFrame({'smiles': smiles_list})
        smi_df.to_csv(smiles_file, index=False)
        all_preds = pd.DataFrame()
        all_attention = pd.DataFrame()
        for task in colnames_dict.keys():
            preds_df, attention_df = tox_predict(task,
                                                 smiles_file,
                                                 pred_file,
                                                 smiles_list
                                                 )
            all_preds = pd.concat([all_preds, preds_df], axis=1)
            all_attention = pd.concat([all_attention, attention_df], axis=1)
        all_preds.index.name = 'smiles'
    rows_to_remove = ["Honey_bee_toxicity", "Honey_bee_toxicity_uncertainty",
                      "LC50_Mallard_Duck", "LC50_Mallard_Duck_uncertainty"]
    all_preds = all_preds.drop(rows_to_remove, axis=1)
    all_attention = all_attention.drop(
        ["Honey_bee_toxicity", "LC50_Mallard_Duck"], axis=1)
    return all_preds, all_attention


def visualize_attention(
    smiles: str,
    atom_weights: torch.FloatTensor,
):
    vir_dir = 'vir.svg'
    with ProcessRunner() as P:
        visualize_atom_attention(vir_dir, smiles, atom_weights)
        with open(vir_dir, 'r') as f:
            return f.read()


if __name__ == '__main__':
    import time
    import pandas as pd
    # smiles_list = ['123', 'CCC', 'CCCC', 'OCC']
    smiles_list = ['CC(C)OC(=O)CC(=O)CSC1=C(C=C2CCCC2=N1)C#N']
    # df = pd.read_csv('test.csv')
    # smiles_list = df['SMILES'].tolist()
    start = time.time()
    # with suppress_stdout_stderr():  # suppress_stdout_stderr() 用于屏蔽 chemprop 的输出
    preds_df, attention_df = main(smiles_list)

    end = time.time()
    # 以上 3 个分子 23 秒
    print('Time:', end-start)
    print(preds_df)
    print(attention_df)

    svg_str = visualize_attention(
        smiles_list[0], attention_df.loc[smiles_list[0], 'hERG_II'])
    print(svg_str)

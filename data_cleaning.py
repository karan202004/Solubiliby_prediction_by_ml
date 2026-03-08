import pandas as pd
import os
from rdkit import Chem
from rdkit.Chem import SaltRemover
from rdkit.Chem.MolStandardize import rdMolStandardize


input_file = r'C:\Users\karan\PycharmProjects\solubility_prediction_ml\aqsoldb_data.csv'

df = pd.read_csv(input_file)
print(df.head())

#droping the drug_id column
df = df[["Drug","Y"]].rename(columns={"Drug":"SMILES","Y":"LogS"})
print(df.head())
print("shape of the dataset=== ",df.shape)

#droping the missing values and duplicates
df = df.dropna(subset=['SMILES','LogS'])
df = df.drop_duplicates("SMILES")
df = df.reset_index(drop=True)


salt_remover = SaltRemover.SaltRemover()
tautomer_enumerator = rdMolStandardize.TautomerEnumerator()

print("Before cleaning")
print(df.head(10))

cleaned_smiles = []
cleaned_logs = []

for index, row in df.iterrows():
    smiles = row['SMILES']
    LogS = row['LogS']

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        continue

    # sanitizing the smiles
    try:
        Chem.SanitizeMol(mol)
    except Exception as e:
        continue

    #remove a salt
    mol = salt_remover.StripMol(mol, dontRemoveEverything=True)

    #tautomer standardization
    mol = tautomer_enumerator.Canonicalize(mol)

    #canonical SMILES
    mol_smiles = Chem.MolToSmiles(mol, canonical=True)

    cleaned_smiles.append(mol_smiles)
    cleaned_logs.append(LogS)

new_df = pd.DataFrame({"SMILES": cleaned_smiles, "LogS" : cleaned_logs })

print("After cleaning")
print(new_df.head(10))

new_df = new_df.drop_duplicates(subset = ["SMILES"]).reset_index(drop=True)

output_file = r'C:\Users\karan\PycharmProjects\solubility_prediction_ml\cleaned_file.csv'

new_df.to_csv(output_file, index = False)

print(new_df.shape)
print(f"cleaned file is saved to a folder {output_file}")




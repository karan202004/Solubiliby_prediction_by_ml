import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors

input_file = r'C:\Users\karan\PycharmProjects\solubility_prediction_ml\cleaned_file.csv'
df = pd.read_csv(input_file)

#collocting the descriptors names
all_descriptors_names = []
for name in Descriptors._descList:
    names = name[0]
    all_descriptors_names.append(names)
print("no of descriptors", len(all_descriptors_names))

cal_descriptors = MoleculeDescriptors.MolecularDescriptorCalculator(all_descriptors_names)

features = []

for i in df['SMILES']:
    mol = Chem.MolFromSmiles(i)
    if mol:
        values = cal_descriptors.CalcDescriptors(mol)
    else:
        values = [None], len(all_descriptors_names)
    features.append(values)

cal_df = pd.DataFrame(features, columns=all_descriptors_names)

#joining of dataframes
new_df = pd.concat([df,cal_df],axis=1)

new_df = new_df.dropna().reset_index(drop=True)

#saving the results
output_file = r'C:\Users\karan\PycharmProjects\solubility_prediction_ml\cal_descriptor_file.csv'

new_df.to_csv(output_file, index=False)
print("output file is saved to ",output_file)
print(new_df.head())
print()
print(new_df.shape)
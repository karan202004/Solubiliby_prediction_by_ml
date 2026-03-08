import pandas as pd
import numpy as np
import joblib
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit.Chem import SaltRemover
from rdkit.Chem.MolStandardize import rdMolStandardize

model_path = r'C:\Users\karan\PycharmProjects\solubility_prediction_ml\solubility_model.pkl'
model = joblib.load(model_path)

expected_features = model.feature_names_in_

all_descriptors_names = []
for name in Descriptors._descList:
    names = name[0]
    all_descriptors_names.append(names)

cal_descriptors = MoleculeDescriptors.MolecularDescriptorCalculator(all_descriptors_names)

salt_remover = SaltRemover.SaltRemover()
tautomer_enumerator = rdMolStandardize.TautomerEnumerator()

def predict_solubility(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return "invalid smiles"

    # sanitizing the smiles
    try:
        Chem.SanitizeMol(mol)
    except:
        return "sanitization failed"

        # remove a salt
    mol = salt_remover.StripMol(mol, dontRemoveEverything=True)

    # tautomer standardization
    mol = tautomer_enumerator.Canonicalize(mol)

    #calculate descriptors
    ds = cal_descriptors.CalcDescriptors(mol)

    df_features = pd.DataFrame([ds], columns=all_descriptors_names)
    df_features = df_features[expected_features]

    df_features = df_features.apply(pd.to_numeric, errors='coerce')
    float32_max = np.finfo(np.float32).max

    if not np.all(np.isfinite(df_features)) or not np.all(np.abs(df_features) <= float32_max):
        return "Error: Molecule produces non-finite values (Overflow)"

    #prediction
    prediction = model.predict(df_features.astype(np.float64))

    return round(prediction[0], 3)


if __name__ == "__main__":
    test_smiles = input("Enter a SMILES string to predict solubility: ")
    result = predict_solubility(test_smiles)

    print(f"Predicted LogS: {result}")

    if isinstance(result, str):
        print("Prediction failed.")
    else:
        if result > -1:
            print("Highly soluble")
        elif -4 < result <= -1:
            print("Moderately soluble")
        else:
            print("Poorly soluble")



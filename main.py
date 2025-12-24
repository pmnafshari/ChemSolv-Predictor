import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

def smile_to_vector(smile):
    mol = Chem.MolFromSmiles(smile)
    if mol is None:
        return None
    # change nBits (2048) to change fingerprint size, radius (2) affects molecular features
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
    return np.array(fp)

def load_and_train():
    url = "https://raw.githubusercontent.com/deepchem/deepchem/master/datasets/delaney-processed.csv"
    data = pd.read_csv(url)
    
    # filter out invalid molecules that fail to convert
    valid_indices = []
    for idx, smile in enumerate(data['smiles']):
        vec = smile_to_vector(smile)
        if vec is not None:
            valid_indices.append(idx)
    
    data = data.iloc[valid_indices]
    X = np.array([smile_to_vector(x) for x in data['smiles']])
    y = data['measured log solubility in mols per litre'].values
    
    # change test_size (0.2 = 20%) to change train/test split ratio
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # change n_estimators (300) to change number of trees, affects accuracy and speed
    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    
    print(f"R2: {r2:.4f}, MSE: {mse:.4f}")
    
    return model, y_test, y_pred

if __name__ == "__main__":
    model, y_test, y_pred = load_and_train()
    
    # change figsize to change plot size
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.6, c='blue', edgecolors='k')
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    plt.xlabel('Actual Solubility (LogS)')
    plt.ylabel('Predicted Solubility (LogS)')
    plt.title('Actual vs Predicted')
    plt.grid(True)
    plt.show()

    # add or remove molecules here to test different compounds
    custom_molecules = {
        "Water": "O",
        "Ethanol": "CCO",
        "Aspirin": "CC(=O)OC1=CC=CC=C1C(=O)O",
        "Benzene": "c1ccccc1",
        "Ibuprofen": "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",
        "Salt (NaCl)": "[Na+].[Cl-]",
        "Glucose": "C(C1C(C(C(C(O1)O)O)O)O)O"
    }

    for name, smile in custom_molecules.items():
        vec = smile_to_vector(smile)
        if vec is not None:
            prediction = model.predict(vec.reshape(1, -1))[0]
            print(f"{name}: {prediction:.4f}")
        else:
            print(f"{name}: Invalid SMILES")
import pandas as pd
import torch
from torch import nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator

def main():

    # Prepare and load the dataset
    data = pd.read_csv("BBBP.csv")

    # Disable RDKit warnings and errors
    RDLogger.DisableLog('rdApp.warning')
    RDLogger.DisableLog('rdApp.error')

    # Remove rows with invalid SMILES strings and create RDKit molecule objects
    data["mol"] = data["smiles"].apply(Chem.MolFromSmiles)
    data_clean = data.dropna(subset=["mol"]).copy()

    # Set up the Morgan fingerprint generator 
    morgan_generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=2, # Consider substructures up to approximately 2 bonds away from each atom
        fpSize=2048 # Generate Morgan fingerprints as 2048-bit vectors
    )

    # Define the classification features by converting the RDKit molecule objects into Morgan fingerprints
    features = []
    for mol in data_clean["mol"]:
        features.append(morgan_generator.GetFingerprint(mol))
  
    # Define the classification target ("0" = does not penetrate, "1" = penetrates)
    target = data_clean["p_np"]

    # Convert the features and target into PyTorch tensors
    features = torch.tensor(features, dtype=torch.float32)  
    target = torch.tensor(target.values, dtype=torch.float32)
    
    # Split the shuffled data into a training set and a test set while preserving class proportions
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.25, random_state=42, stratify=target)

    # Build a simple, 2-layer, feed-forward neural network model using nn.Sequential
    classification_model = nn.Sequential(
        nn.Linear(in_features=2048, out_features=128), # First layer takes the 2048-bit fingerprint as input and outputs 128 features
        nn.ReLU(), # Allows the model to learn more complicated, non-linear relationships 
        nn.Linear(in_features=128, out_features=1) # Second layer takes the 128 features and outputs a single value (a logit)
    )

    # Define the loss function and optimizer
    loss_function = nn.BCEWithLogitsLoss() #BCEWithLogitsLoss is used since the target is binary and the model has a single output
    optimizer = optim.Adam(classification_model.parameters(), lr=0.001) # Learning rate controls how much the model is changed in response to each estimated error during training

    # Train the neural network model
    classification_model.train() # Put the model into training mode
    for _ in range(100): # Repeat the training process for 100 epochs (complete passes through the training data)
        optimizer.zero_grad() # Clear the gradients from the previous epoch to prevent accumulation
        logits = classification_model(X_train).reshape(-1) # Use the model to make raw predictions
        loss = loss_function(logits, y_train) # Calculate the loss by comparing the predictions to the true classes
        loss.backward() # Use backpropagation to calculate a gradient for each parameter; a measure of how any changes in the weight affect the loss
        optimizer.step() # Change the model's weights based on gradients calculated during backpropagation

    # Use the model to make predictions on the test data
    classification_model.eval() # Put the model into evaluation mode
    with torch.no_grad(): # Specify that the model shouldn't calculate or store any gradients
        logits = classification_model(X_test) # Use the model to make raw predictions on the test set 
        probabilities = torch.sigmoid(logits) # Convert raw predictions to probabilities
        predictions = (probabilities >= 0.5).float() # Convert probabilities to binary classes

    # Use the predictions on the test set to determine the model's accuracy
    print(f"Overall Accuracy: {accuracy_score(y_test, predictions):.2%}")
    print(f"ROC-AUC Score: {roc_auc_score(y_test, probabilities):.4f}\n")
    print(classification_report(y_test, predictions, labels=[0,1], target_names=["Does not penetrate", "Penetrates"]))

    # Obtain data for a new molecule
    new_smiles = input("Enter SMILES string: ")
    new_mol = Chem.MolFromSmiles(new_smiles)

    # Predict whether the new molecule penetrates the blood-brain barrier
    if new_mol is not None:
        molecule = morgan_generator.GetFingerprint(new_mol)
        molecule = torch.tensor(molecule, dtype=torch.float32)  
        with torch.no_grad():
            logit = classification_model(molecule.reshape(1, -1))
            probability = torch.sigmoid(logit)
        if probability < 0.5:
            print("Does not penetrate the blood-brain barrier.")
        else:
            print("Penetrates the blood-brain barrier.")
    else:
        print("Invalid SMILES string.")

if __name__ == "__main__":
    main()


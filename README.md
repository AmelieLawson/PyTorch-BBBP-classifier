# PyTorch Blood-Brain Barrier Penetration Classifier

## Project Overview

The aim of this project was to train a two-layer, feed-forward neural network to predict if a molecule would penetrate the blood-brain barrier based on its SMILES string.

## Dataset

The dataset used in this project is the BBBP (Blood–Brain Barrier Penetration) dataset, from MoleculeNet/DeepChem, originally described by Martins et al. (2012). It is not included in this repository. Place `BBBP.csv` in the same directory as `bbbp.py` before running the program.

[BBBP Dataset – DeepChem](https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/BBBP.csv?utm_source=chatgpt.com)

The dataset contains the names of 2,050 molecules, their SMILES strings, and whether or not they penetrate the blood-brain barrier.

## Method

The steps involved in the project are outlined below:

1. Load the dataset using pandas.
2. Disable RDKit warnings and errors.
3. Use SMILES strings to create RDKit molecule objects.
4. Remove rows with invalid SMILES strings (i.e. instances where the RDKit molecule object is `None`, of which there were 11).
5. Set up the Morgan fingerprint generator.
6. Convert the RDKit molecule objects into Morgan fingerprints and define as the features used for classification.
7. Define blood-brain barrier permeability as the target variable for classification.
8. Convert the features and target into PyTorch tensors
9. Split the data into a training set and a test set using a 75/25 split.
10. Build a simple, two-layer, feed-forward neural network.
11. Define the loss function and optimizer.
12. Train the neural network using the training data.
13. Evaluate the model's performance on the test data using its accuracy, ROC-AUC score, and classification report.
14. Allow the user to enter the SMILES string of a new molecule and predict whether it penetrates the blood-brain barrier.

## Evaluation

The metrics used to evaluate the model were as follows:

- Accuracy: The proportion of test predictions which were correct.
- ROC/AUC: How well the model distinguishes between the two classes.
- Classification report: Includes precision (e.g. the proportion of molecules predicted to penetrate the blood-brain barrier which actually do penetrate the blood-brain barrier), recall (e.g. the proportion of molecules which penetrate the blood-brain barrier which were correctly predicted to penetrate the blood-brain barrier), and F1 score (precision and recall combined into a single metric) for each class.

## How to Run

Install the required Python libraries:

```bash
pip install pandas scikit-learn rdkit torch
```

## What I Learnt

Below is a summary of what I learnt while completing this project:

- How to convert from pandas DataFrames to PyTorch tensors.
- How to use nn.Sequential to build a simple, feed-forward neural network.
- The difference between a logit and a probability.
- How to define a loss function and an optimizer.
- The significance of the learning rate, as defined within the optimiser (controls how much the model is changed in response to each estimated error during training).
- How to train a simple, feed-forward neural network, including the purpose of each of the steps involved.
- How to use torch.sigmoid to convert from logins to binary classes.

## Additional Considerations



When splitting the data into a training and a test

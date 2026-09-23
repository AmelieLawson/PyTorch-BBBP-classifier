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
8. Convert the features and target into PyTorch tensors.
9. Split the data into a training set and a test set using a 75/25 split.
10. Build a simple, two-layer, feed-forward neural network using `nn.Sequential`.
11. Define the loss function and optimizer.
12. Train the neural network using the training data.
13. Evaluate the model's performance on the test data (using accuracy, ROC-AUC score, and a classification report).
14. Allow the user to enter the SMILES string of a new molecule and predict whether it penetrates the blood-brain barrier.

## Evaluation

The metrics used to evaluate the model were as follows:

- Accuracy: The proportion of test predictions which were correct.
- ROC/AUC: How well the model distinguishes between the two classes.
- Classification report: Includes precision (e.g. the proportion of molecules predicted to penetrate the blood-brain barrier which actually do penetrate the blood-brain barrier), recall (e.g. the proportion of molecules which penetrate the blood-brain barrier which were correctly predicted to penetrate the blood-brain barrier), and F1 score (the balance between precision and recall) for each class.

## How to Run

Install the required Python libraries:

```bash
pip install pandas scikit-learn rdkit torch
```

## What I Learnt

Below is a summary of what I learnt while completing this project:

- What is meant by a feed-forward neural network (a neural network in which information moves in a single direction from input to output when making a prediction).
- How to convert lists and pandas Series to PyTorch tensors.
- How to use `nn.Sequential` to build a simple, feed-forward neural network.
- The benefit of introducing non-linearity into the model (allows model to learn more complex relationships between variables and hence represent more complicated patterns in the training data).
- The difference between a logit and a probability (a logit is the raw output of the model and can be any real number, its corresponding probability is a number between 0 and 1 determined by feeding the logit into the `torch.sigmoid` function).
- How to define a loss function and an optimizer.
- How to choose the correct loss function for binary classification depending on whether the neural network outputs a single logit (`BCEWithLogitsLoss`) or two separate logits, one per class (`CrossEntropyLoss`).
- That there are multiple different types of optimizer, and that Adam keeps track of the history of the gradients for each parameter so it can automatically adjust how much each individual weight should be changed.
- The significance of the learning rate, as defined within the optimizer (controls how much the model is changed in response to each estimated error during training).
- How to train a simple, feed-forward neural network, including the purpose of each of the steps involved.
- What is meant by an epoch (a single pass through the entire training data).
- How to use `torch.sigmoid` to convert from a logit to a probability, and how to use probability to predict a binary class.
- The meaning of the line ` with torch.no_grad()` (to tell the model that, unlike during training, it shouldn't calculate or store gradients, it should just make predictions).

## Areas for Advancement

- Try using `nn.Sequential` to build a neural network with more than two layers.
- Try using `nn.Sequential` to build a neural network for a binary classification problem which outputs two logits instead of one.
- Try using `nn.Sequential` to build a neural network for a multi-class classification problem, or a regression problem.
- Try building a neural network using `nn.Module` instead, and develop a better understanding of how this differs from `nn.Sequential`.
- Learn more about the other optimizers and the conditions in which they would be used.
- Learn more about how "training mode" and "evaluation mode" differ, and why they are important to clarify.
- Develop a better overall understanding of neural networks so I can identify my own misconceptions and hopefully spot any errors in my explanations above.



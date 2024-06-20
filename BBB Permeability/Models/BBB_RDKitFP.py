# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 11:09:29 2024

@author: marta
"""

import pandas as pd
import numpy as np
from rdkit import Chem
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, confusion_matrix, accuracy_score, f1_score, precision_score, matthews_corrcoef
from imblearn.over_sampling import SMOTE
from collections import Counter
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.decomposition import KernelPCA
import warnings

# Function to convert SMILES to RDKit fingerprint
def smiles_to_rd_fp(smiles, fpSize=1024):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        # Remove implicit hydrogens
        mol = Chem.RemoveHs(mol)
        
        # Generate RDKit fingerprint
        rd_fp = Chem.RDKFingerprint(mol, fpSize=fpSize)
        return np.array(rd_fp)
    except Exception as e:
        # If an error occurs during molecule creation, return None
        return None

# Function to train model
def train_model(X_train, y_train, X_val, y_val):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(4000, activation='relu', input_shape=(X_train.shape[1],)),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(2000, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(1000, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(500, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(250, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True)
    history = model.fit(X_train, y_train, epochs=30, batch_size=64, validation_data=(X_val, y_val), callbacks=[early_stopping], verbose=1)
    return model, history

# Function to evaluate model
def evaluate_model(model, X_test, y_test):
    loss, accuracy = model.evaluate(X_test, y_test)
    y_pred = model.predict(X_test)
    auc_roc = roc_auc_score(y_test, y_pred)
    precision, recall, _ = precision_recall_curve(y_test, y_pred)
    auc_prc = auc(recall, precision)
    y_pred_class = np.round(y_pred)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_class).ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    mcc = matthews_corrcoef(y_test, y_pred_class)
    fpr = fp / (fp + tn)
    fnr = fn / (fn + tp)
    f1 = f1_score(y_test, y_pred_class)
    precision = precision_score(y_test, y_pred_class)
    return loss, accuracy, auc_roc, auc_prc, sensitivity, specificity, mcc, fpr, fnr, f1, precision

# Function to reduce dimensionality using KPCA
def reduce_dimensionality(X_train, X_val, X_test, n_components=100):
    kpca = KernelPCA(n_components=n_components, kernel='rbf')
    X_train_kpca = kpca.fit_transform(X_train)
    X_val_kpca = kpca.transform(X_val)
    X_test_kpca = kpca.transform(X_test)
    return X_train_kpca, X_val_kpca, X_test_kpca

# Read data from CSV file
file_path = "C:/Users/marta/OneDrive/Backup/Ambiente de Trabalho/Universidade/Licenciatura/3ºano/2º Semestre/Projeto em Engenharia Biomédica/BBB Permeability/Data/data_bbb_all.csv"

try:
    df = pd.read_csv(file_path)
    df = df.dropna(subset=['SMILES'])
    df['SMILES'] = df['SMILES'].astype(str)
    smiles = df["SMILES"].tolist()
    bbb = df["bbb"].tolist()
    ecfps = np.array([smiles_to_rd_fp(smile, fpSize=1024) for smile in smiles if smiles_to_rd_fp(smile, fpSize=1024) is not None])
    bbb = np.array(bbb)[:ecfps.shape[0]]
    warnings.filterwarnings("ignore")

    # Apply SMOTE
    sm = SMOTE(random_state=42)
    ecfps_res, bbb_res = sm.fit_resample(ecfps, bbb)
    print('Original dataset shape %s' % Counter(bbb))
    print('Resampled dataset shape %s' % Counter(bbb_res))

    X_train, X_test, y_train, y_test = train_test_split(ecfps_res, bbb_res, test_size=0.15, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1765, random_state=42)  # Validation split on remaining data after train split

    # Reduce dimensionality using KPCA
    n_components = 100  # You can adjust this parameter as needed
    X_train_kpca, X_val_kpca, X_test_kpca = reduce_dimensionality(X_train, X_val, X_test, n_components)

    model, history = train_model(X_train_kpca, y_train, X_val_kpca, y_val)
    loss, accuracy, auc_roc, auc_prc, sensitivity, specificity, mcc, fpr, fnr, f1, precision = evaluate_model(model, X_test_kpca, y_test)

    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Model Accuracy')
    plt.legend()
    plt.show()

    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Model Loss')
    plt.legend()
    plt.show()

    print(f"Test Loss: {loss}\nTest Accuracy: {accuracy}\nROC-AUC: {auc_roc}\nPRC-AUC: {auc_prc}")
    print(f"Sensitivity: {sensitivity}\nSpecificity: {specificity}")
    print(f"Matthews Correlation Coefficient: {mcc}")
    print(f"False Positive Rate: {fpr}\nFalse Negative Rate: {fnr}")
    print(f"F1 Score: {f1}\nPrecision: {precision}")

except FileNotFoundError:
    print(f"The file '{file_path}' was not found.")
except Exception as e:
    print(f"An error occurred: {e}")
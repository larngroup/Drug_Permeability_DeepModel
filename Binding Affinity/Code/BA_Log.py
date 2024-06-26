# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 14:23:43 2024

@author: marta
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dense, GRU, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr
from sklearn.preprocessing import FunctionTransformer

# Function to tokenize SMILES strings and count character frequencies
def smiles_tokenizer(smiles_list):
    special_tokens = ['Cl', 'Br', 'Si', 'Na', 'Se']
    tokens = {}
    
    for smile in smiles_list:
        i = 0
        while i < len(smile):
            if smile[i] == '[':
                token_end = smile.find(']', i) + 1
                token = smile[i:token_end]
                i = token_end
            elif i < len(smile) - 1 and smile[i:i+2] in special_tokens:
                token = smile[i:i+2]
                i += 2
            else:
                token = smile[i]
                i += 1
            
            if token in tokens:
                tokens[token] += 1
            else:
                tokens[token] = 1
    
    token_dict = {token: idx for idx, token in enumerate(sorted(tokens.keys()), start=1)}
    token_dict[' '] = 0
    
    return token_dict, tokens

# Function to encode SMILES strings using the token dictionary
def encode_smiles(smiles_list, token_dict):
    smiles_encoded = []
    for smile in smiles_list:
        smile_encoded = []
        i = 0
        while i < len(smile):
            if smile[i] == '[':
                token_end = smile.find(']', i) + 1
                token = smile[i:token_end]
                i = token_end
            elif i < len(smile) - 1 and smile[i:i+2] in token_dict:
                token = smile[i:i+2]
                i += 2
            else:
                token = smile[i]
                i += 1
            smile_encoded.append(token_dict[token])
        smiles_encoded.append(smile_encoded)
    return smiles_encoded

# Function to read and preprocess data from the specified file path
def read_smiles_from_csv(file_path):
    df = pd.read_csv(file_path)
    
    # Calculate percentiles
    percentile_05 = np.percentile(df['pIC50'], 5)
    percentile_95 = np.percentile(df['pIC50'], 95)
    
    # Cut data at 5th and 95th percentiles
    df_cut = df[(df['pIC50'] >= percentile_05) & (df['pIC50'] <= percentile_95)]
    
    smiles_list = df_cut['SMILES'].tolist()
    y = df_cut['pIC50'].values
    
    # Logarithmic normalization for the target variable
    log_transformer = FunctionTransformer(np.log1p, np.expm1)
    y_log = log_transformer.fit_transform(y.reshape(-1, 1)).flatten()
    
    return smiles_list, y_log, log_transformer

# Function for creating the model using provided specifications
def model_creation(token_dict_length):
    Dropout = 0.3
    Activation = 'relu'
    InputLength = 65
    unitsEmbedding = 128
    unitsGRU = 128
    unitsDense = 128
    unitsOutput = 1

    model = Sequential()

    model.add(Input(shape=(InputLength,)))

    model.add(Embedding(input_dim=token_dict_length, output_dim=unitsEmbedding, input_length=InputLength))

    model.add(GRU(units=unitsGRU, return_sequences=True, dropout=Dropout))
    model.add(GRU(units=unitsGRU, dropout=Dropout))

    model.add(Dense(units=unitsDense, activation=Activation))

    model.add(Dense(units=unitsOutput, activation='linear'))

    print(model.summary())

    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

    return model

# Function to calculate Concordance Correlation Coefficient (CCC)
def concordance_correlation_coefficient(y_true, y_pred):
    mean_true = np.mean(y_true)
    mean_pred = np.mean(y_pred)
    var_true = np.var(y_true)
    var_pred = np.var(y_pred)
    sd_true = np.std(y_true)
    sd_pred = np.std(y_pred)
    pearson_corr = pearsonr(y_true, y_pred)[0]
    numerator = 2 * pearson_corr * sd_true * sd_pred
    denominator = var_true + var_pred + (mean_true - mean_pred)**2
    return numerator / denominator

# Function to train and evaluate the model using K-Fold cross-validation
def train_evaluate_model(X_train_val, y_train_val, X_test, y_test, token_dict_length, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    mse_scores, r2_scores, ccc_scores = [], [], []
    y_test_preds = np.zeros_like(y_test)
    best_run_params = {'mse': float('inf'), 'r2': -float('inf'), 'ccc': -float('inf'), 'preds': None}

    for train_index, val_index in kf.split(X_train_val):
        X_train, X_val = X_train_val[train_index], X_train_val[val_index]
        y_train, y_val = y_train_val[train_index], y_train_val[val_index]
        
        model = model_creation(token_dict_length)
        
        # Model checkpoint callback
        checkpoint_path = "best_model_fold.keras"
        checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_loss', save_best_only=True, mode='min', verbose=1)
        
        # Early stopping callback
        early_stopping = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)

        model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_val, y_val), verbose=1, callbacks=[checkpoint, early_stopping])

        # Load the best model saved by ModelCheckpoint
        model.load_weights(checkpoint_path)

        # Evaluate on the test set for each fold       
        y_pred_log = model.predict(X_test).flatten()
        y_test_preds += y_pred_log
        
        mse = mean_squared_error(y_test, y_pred_log)
        r2 = r2_score(y_test, y_pred_log)
        ccc = concordance_correlation_coefficient(y_test, y_pred_log)
        
        mse_scores.append(mse)
        r2_scores.append(r2)
        ccc_scores.append(ccc)
        
        # Update the best run parameters
        if mse < best_run_params['mse']:
            best_run_params.update({'mse': mse, 'r2': r2, 'ccc': ccc, 'preds': y_pred_log})
    
    mean_mse = np.mean(mse_scores)
    mean_r2 = np.mean(r2_scores)
    mean_ccc = np.mean(ccc_scores)
    
    y_test_preds /= n_splits
    
    return mean_mse, mean_r2, mean_ccc, y_test_preds, best_run_params

# File path to the CSV containing SMILES strings
file_path = "C:/Users/marta/OneDrive/Backup/Ambiente de Trabalho/Universidade/Licenciatura/3ºano/2º Semestre/Projeto em Engenharia Biomédica/Binding Affinity/Data/jak2_data.csv"

# Read SMILES strings and target values from the CSV file
smiles_list, y_log, log_transformer = read_smiles_from_csv(file_path)

# Tokenize the SMILES strings and count character frequencies
token_dict, token_counts = smiles_tokenizer(smiles_list)

# Encode the SMILES strings using the token dictionary
smiles_encoded = encode_smiles(smiles_list, token_dict)

# Pad the encoded SMILES strings to ensure they all have the same length (max length = 65)
max_length = 65
smiles_padded = pad_sequences(smiles_encoded, maxlen=max_length, padding='post', value=0)

# Convert padded sequences and scaled target values to numpy arrays
X = np.array(smiles_padded)
y_log = np.array(y_log)

# Split the data into 85% for training/validation and 15% for testing
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y_log, test_size=0.15, random_state=42)

# Train and evaluate the model using K-Fold cross-validation
mean_mse, mean_r2, mean_ccc, y_test_preds, best_run_params = train_evaluate_model(X_train_val, y_train_val, X_test, y_test, len(token_dict))

# Calculate and print Q²
q2_score = 1 - (np.sum((y_test - y_test_preds)**2) / np.sum((y_test - np.mean(y_test))**2))

print(f'Average Mean Squared Error (MSE) across 5 folds: {mean_mse}')
print(f'Average Coefficient of Determination (R²) across 5 folds: {mean_r2}')
print(f'Average Concordance Correlation Coefficient (CCC) across 5 folds: {mean_ccc}')
print(f'Q²: {q2_score}')
print(f'Best Run Parameters: MSE: {best_run_params["mse"]}, R²: {best_run_params["r2"]}, CCC: {best_run_params["ccc"]}')

# Inverse transform the predictions to get them back to the original scale
y_test_preds_original_scale = log_transformer.inverse_transform(best_run_params['preds'].reshape(-1, 1)).flatten()
y_test_original_scale = log_transformer.inverse_transform(y_test.reshape(-1, 1)).flatten()

# Plot the scatter diagram of predicted vs actual values in the original scale
plt.figure(figsize=(10, 6))
plt.scatter(y_test_original_scale, y_test_preds_original_scale, alpha=0.6)
plt.plot([4, 10], [4, 10], 'r--')
plt.xlim([4, 10])
plt.ylim([4, 10])
plt.xlabel('Actual pIC50')
plt.ylabel('Predicted pIC50')
plt.title('Actual vs Predicted pIC50')
plt.show()

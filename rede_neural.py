import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
import os
import warnings
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import plot_model
from tabulate import tabulate
import random
import requests
import traceback
import joblib

def buscar_dados_reais(url_api, nome_arquivo="dados_reais.xlsx"):
    try:
        resposta = requests.get(url_api)
        resposta.raise_for_status()

        dados_json = resposta.json()
        df = pd.DataFrame(dados_json)

        df.columns = ['Id', 'Nome', 'Setor', 'Gender', 'Work-Life Balance', 'Marital Status', 'Job Level', 'Remote Work', 'Prob_Permanencia', 'Attrition', 'Created at']

        if 'Gender' in df.columns:
            df['Gender'] = df['Gender'].astype(str).str.strip().str.lower().replace({'feminino': 0, 'masculino': 1}).infer_objects(copy=False)

        if 'Work-Life Balance' in df.columns:
            df['Work-Life Balance'] = df['Work-Life Balance'].astype(str).str.strip().str.lower().replace({
                'razoavel': 0, 'ruim': 1, 'bom': 2, 'excelente': 3
            }).infer_objects(copy=False)

        if 'Marital Status' in df.columns:
            df['Marital Status'] = df['Marital Status'].astype(str).str.strip().str.lower().replace({
                'solteiro': 0, 'casado': 1, 'divorciado': 2
            }).infer_objects(copy=False)

        if 'Job Level' in df.columns:
            df['Job Level'] = df['Job Level'].astype(str).str.strip().str.lower().replace({
                'entry': 0, 'mid': 1, 'senior': 2
            }).infer_objects(copy=False)

        if 'Remote Work' in df.columns:
            df['Remote Work'] = df['Remote Work'].astype(str).str.strip().str.lower().replace({
                'nao': 0, 'sim': 1
            }).infer_objects(copy=False)

        df.to_excel(nome_arquivo, index=False, engine='openpyxl')
        print(f"Dados salvos com sucesso em: {nome_arquivo}")
        return df

    except Exception as e:
        print("Erro ao buscar ou salvar os dados:")
        traceback.print_exc()
        return None


def carregar_dados(caminho_train, caminho_test):
    data_train = pd.read_csv(caminho_train, sep=';', encoding='utf-8')
    data_test = pd.read_csv(caminho_test, sep=';', encoding='utf-8')
    data = pd.concat([data_train, data_test], ignore_index=True)
    return data

def preprocessar_dados(data):
    data['Attrition'] = data['Attrition'].replace({'Left': 1, 'Stayed': 0}).astype(int)
    data_filtrado = data[['Gender', 'Work-Life Balance', 'Marital Status', 'Job Level', 'Remote Work', 'Attrition']]
    data_filtrado = pd.get_dummies(data_filtrado, drop_first=True)
    return data_filtrado

def dividir_dados(data_filtrado):
    X = data_filtrado.drop('Attrition', axis=1)
    y = data_filtrado['Attrition']
    return train_test_split(X, y, test_size=0.3, random_state=42)

def normalizar_dados(X_train, X_test):
    scaler = MinMaxScaler()
    X_train_n = scaler.fit_transform(X_train)
    X_test_n = scaler.transform(X_test)

    joblib.dump(scaler, 'scaler_attrition.save')

    return pd.DataFrame(X_train_n, columns=X_train.columns), pd.DataFrame(X_test_n, columns=X_test.columns)

def set_seed(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)

def construir_modelo(input_dim):
    model = Sequential([
        Dense(64, input_dim=input_dim, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        Dense(32, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        Dense(16, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])
    model.compile(loss='binary_crossentropy', optimizer='ADAM', metrics=['accuracy', 'AUC'])
    return model

def treinar_modelo(model, X_train, y_train):
    print('\nTreinando o Modelo.\n')
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    history = model.fit(X_train, y_train, epochs=100, batch_size=64, validation_split=0.3, callbacks=[early_stopping])
    return history

def avaliar_modelo(model, X_test, y_test):
    loss, accuracy, auc = model.evaluate(X_test, y_test)
    print(f'\nAcurácia no teste: {accuracy:.2f}')
    return loss, accuracy, auc


def main():

    buscar_dados_reais("http://localhost:5000//cadastroFuncionarios/consulta")

    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
    warnings.filterwarnings('ignore')

    set_seed(42)

    data = carregar_dados('train-atualizada.csv', 'test-atualilzada.csv')
    data_filtrado = preprocessar_dados(data)

    X_train, X_test, y_train, y_test = dividir_dados(data_filtrado)
    
    X_train_norm, X_test_norm = normalizar_dados(X_train, X_test)
    
    model = construir_modelo(X_train_norm.shape[1])
    history = treinar_modelo(model, X_train_norm, y_train)
    avaliar_modelo(model, X_test_norm, y_test)
    
    y_pred_prob = model.predict(X_test_norm)

    model.save("modelo_attrition.keras") 
    


if __name__ == "__main__":
    main()

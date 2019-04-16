import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from metricas import mape
from preprocessamento import *
from preditores import *





def dados(serie_name):
    '''
    Entrada: Nome da série
    Saída: Dicionario com dados e informações    
    '''   
    print('Série:', serie_name)
    endereco = 'series/'+serie_name+'.txt'
    dados = pd.read_csv(endereco, delimiter=' ', header=None)
    serie = dados[0]
    serie_normalizada = normalise(serie)
    train, test = split_serie_less_lags(serie_normalizada, 0.75)
    max_lag = 20
    lags_acf = select_lag_acf(serie_normalizada, max_lag)
    max_sel_lag = lags_acf[0]

    train_lags = create_windows(train, max_sel_lag+1)

    test_lags = create_windows(test, max_sel_lag+1)


    tam_val = int(len(train_lags)*0.32)

    train_data,  val_data = select_validation_sample(train_lags, 0.34)
    info_dados = {'nome': serie_name, 'serie': serie, 'serie_normalizada': serie_normalizada, 'lags_acf': lags_acf, 'train_lags': train_data, 'test_lags':test_lags, 'tam_val': tam_val, 'val_data': val_data}

    return info_dados



def train_model(x_train, y_train, x_val, y_val, lags_acf, ex=5, iter=20):
    
    neuronios = [1, 10, 20, 50,  100]
    func_opt = ['adam','sgd', 'rmsprop']
    best_result = np.Inf

    for i in range(0,len(neuronios)):
        for j in range(0,len(func_opt)):
            lstm = gerar_lstm(neuronios[i], len(lags_acf), func_opt = func_opt[j])
            lstm_treinada, mse_val = train_lstm(lstm, x_train, y_train, x_val, y_val, num_ex=ex,epochs=iter)
            if mse_val < best_result:
                best_result = mse_val
                select_model = lstm_treinada
                print('melhor configuração. neuronios:', neuronios[i], 'funcao:', func_opt[j])
    return select_model 


def run():
    

    series_nomes = ['carsales']

    for s in series_nomes:
        print('Série Executando:', s)
    
        serie = s
        
        inf_dados = dados(serie)
        tam_val = inf_dados['tam_val']
        train_lags = inf_dados['train_lags']
        val_data = inf_dados['val_data']
        lags_acf = inf_dados['lags_acf']
        
        x_train = train_lags[:,0:-1]
        x_train = x_train[:,lags_acf]  #retorna só os lags selecionados no acf
        y_train = train_lags[:,-1]
        x_val = val_data[:,0:-1]
        x_val = x_val[:,lags_acf]
        y_val = val_data[:,-1]     
   


        modelo_lstm = train_model(x_train, y_train, x_val, y_val,lags_acf, ex=5, iter=20)

        filename = serie+'_LSTM.h5'
        modelo_lstm.save(filename)



run()
       
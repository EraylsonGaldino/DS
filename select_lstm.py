import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils


from sklearn.externals import joblib
from metricas import mape, smape, mse, mae, pocid
import matplotlib.pyplot as plt
from select_approach import *
from preprocessamento import *  
import pandas as pd


def dados(serie_name):
    '''
    Entrada: Nome da série
    Saída: Dicionario com dados e informações    
    '''
    


   
    print('Série:', serie_name)
    endereco = '/series/'+serie_name+'.txt'
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
    info_dados = {'nome': serie_name, 'serie': serie, 'serie_normalizada': serie_normalizada, 'lags_acf': lags_acf, 'train_lags': train_lags, 'test_lags':test_lags, 'tam_val': tam_val}

    return info_dados



def carregar_pool(serie_name, iter=20, exe=10):
    full_ensemble = {'modelos':[], 'tam_lags':[], 'indices':[]}
    name_pool = serie_name+'_full_pool_100.sav'
    loaded_ensemble = joblib.load(name_pool)
    full_ensemble['modelos'] = loaded_ensemble['modelos']
    full_ensemble['tam_lags'] = loaded_ensemble['tam_lags']
    full_ensemble['indices']  = loaded_ensemble['indices']
    tam = len(full_ensemble['modelos'])
    print('Total do Ensemble: ', tam)
    return full_ensemble


def treinar_lstm(x_train, y_train, x_val, y_val, exe=10, iter=50):

    lstm_neuronios = [5, 10, 50, 100, 500]
    func_act = ['sigmoid', 'relu']


    X_tr = numpy.reshape(x_train,(x_train.shape[0], x_train.shape[1], 1))
    X_v = numpy.reshape(x_val,(x_val.shape[0], x_val.shape[1], 1))
    Y_tr = y_train
    Y_v = y_val


    best_result = np.Inf

    for ln in lstm_neuronios: 
        for fc in func_act: 
            for i in range(0, exe):
                        
                # define the LSTM model
                model = Sequential()
                model.add(LSTM(ln, input_shape=(X_tr.shape[1], X_tr.shape[2])))
                #model.add(Dropout(0.2))
                model.add(Dense(Y_tr.shape[1], activation=fc))
                #model.compile(loss='categorical_crossentropy', optimizer='adam')
                model.compile(loss='mean_squared_error', optimizer='adam')
                model.fit(X_tr, Y_tr, epochs=iter, batch_size=Y_tr.shape[0], verbose=0)

                prev_val = model.predict(X_v)
                mse_val = mse(Y_v, prev_val)
                if mse_val < best_result:
                    best_result = mse_val
                    melhor_modelo = model
                    best_fc = fc
                    best_ln = ln
                    print('Neuronios: ', ln, ' Função:', fc)
                    print('MSE', best_result)
    

    return melhor_modelo
        



def run():
    

    series_nomes = ['pollutions', 'goldman', 'sp', 'wine']

    for s in series_nomes:
        print('Série Executando:', s)
    
        serie = s
        
        inf_dados = dados(serie)
        tam_val = inf_dados['tam_val']
        train_lags = inf_dados['train_lags']
        
        x_train = train_lags[0:-tam_val,0:-1] #example [0,1,2]
        y_train = train_lags[0:-tam_val]   # exemple [0,1,2,3]
        x_val = train_lags[-tam_val:,0:-1]
        y_val = train_lags[-tam_val:]


        modelo_lstm = treinar_lstm(x_train, y_train, x_val, y_val)


        filename = serie+'_LSTM_generator.h5'
        modelo_lstm.save(filename)





run()


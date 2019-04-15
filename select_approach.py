from sklearn.metrics import mean_squared_error as MSE
import numpy as np
from similaridades import *

def selec_model_ola(window_test_inst, past_data, ensemble, k, lags_acf, metrica='euclidian'):
    #  Seleciona o modelo baseado no desempenho obtido em prever as k janelas mais próximas da nova janela 
    
    from scipy.spatial.distance import euclidean 
    
    #max_lag = len(lags_acf)
    x_data = past_data[:,lags_acf]
    y_data = past_data[:,-1]
    
    

    tam = len(x_data[0]) 
    
    dist = []
    for i in range(0,len(x_data)):
        #d = euclidean(window_test_inst, x_data[i,:])
        d = measure_distance(window_test_inst, x_data[i,:], metrica)
        #print(d)
        dist.append(d)
        
    indices_patterns = range(0, len(x_data))
    
    dist, indices_patterns = zip(*sorted(zip(dist, indices_patterns))) #returna tuplas ordenadas
    #print(indices_patterns)
    indices_patterns_l = list(indices_patterns)
 
    k_patterns_x = x_data[indices_patterns_l[0:k]]
    k_patterns_y = y_data[indices_patterns_l[0:k]]
    
    
    
    best_result = np.Inf
    for i in range(0, len(ensemble['modelos'])):
        model = ensemble['modelos'][i]
        lags = ensemble['tam_lags'][i]
        current_patterns = k_patterns_x[:,0:lags+1]
        
        
        prev = model.predict(current_patterns)
        mse = MSE(k_patterns_y, prev)
        #print('MSE do modelo', i,':', mse)     
        
        if mse < best_result:
            best_result = mse
            select_model = model
            ind_best = i
            #print('-------------- Select DS------',i)
    
    return select_model, ind_best




def k_janela_proxima(x_data,  window_test_inst, k):
    from scipy.spatial.distance import euclidean, cityblock 
    dist = []
    for i in range(0,len(x_data)):
        #import pdb; pdb.set_trace()
        d = euclidean(window_test_inst, x_data[i,:])
        
        
        #print(d)
        dist.append(d)
        
    indices_patterns = range(0, len(x_data))
    
    dist, indices_patterns = zip(*sorted(zip(dist, indices_patterns))) #returna tuplas ordenadas
    #print(indices_patterns)
    indices_patterns_l = list(indices_patterns)
   
  
    k_patterns_x = x_data[indices_patterns_l[0:k]]
    
    
    return k_patterns_x, indices_patterns_l[0:k]



def selec_model_best_after(previous_data, ensemble,lags_acf):

    '''
    Seleciona o modelo baseado em quem foi melhor nos pontos anteriores: BPM

    '''


    error = []
    #max_lag = len(lags_acf)
    x_data = previous_data[:, lags_acf]
    y_data = previous_data[:,-1]
    
    #print('@@@@@@@@@@', len(x_data))
    for i in range(0, len(ensemble['modelos'])):
        model = ensemble['modelos'][i]
        lag_model = ensemble['tam_lags'][i]
        
        pattern_previous = x_data[:,0:lag_model+1]
        
        prev = model.predict(pattern_previous)
        er = 0
        #import pdb; pdb.set_trace()
        for j in range(0, len(prev)):
            er = er + np.absolute(y_data[j] - prev[j])
            
        
        
        error.append(er)
    
    
    e = np.Inf
    for i in range(0, len(error)):
        if error[i]< e:
            e = error[i]
            model_select = ensemble['modelos'][i]
            ind_model = i
            
                  
                  
    return model_select, ind_model


def erro_test_models(previsoes_ensemble, target):

    ''' 
    erro dos modelos no ponto anterior
    '''


    erros = []
    for i in range(0, len(previsoes_ensemble)):
        erro = np.absolute(target - previsoes_ensemble[i])
        erros.append(erros)       
        
    return erros 


def k_janela_proxima_dist(x_data,  window_test_inst, k):
    #retorna a soma de distancia dos k padroes
    from scipy.spatial.distance import euclidean, cityblock 
    dist = []
    for i in range(0,len(x_data)):
        #import pdb; pdb.set_trace()
        d = euclidean(window_test_inst, x_data[i,:])       
        
        #print(d)
        dist.append(d)
        
    indices_patterns = range(0, len(x_data))
    
    dist, indices_patterns = zip(*sorted(zip(dist, indices_patterns))) #returna tuplas ordenadas
    #print(indices_patterns)
    indices_patterns_l = list(indices_patterns)
    dist_l = list(dist)
  
    k_patterns_x = x_data[indices_patterns_l[0:k]]
    k_distancias = np.sum(dist_l[0:k])
    
    return k_patterns_x, indices_patterns_l[0:k], k_distancias



def erro_models_train(window_test_inst, x_data_train, y_data_train, full_ensemble,tam_val, k):
    all_err = []
    '''
      erro dos modelos nos k padrões de treinamento de cada modelo mais próx do novo padrão
    
    '''
    for m in range(0, len(full_ensemble['modelos'])):
        indices_treinamento = full_ensemble['indices'][m][0:tam_val]
        x_tr = x_data_train[indices_treinamento]
        y_tr = y_data_train[indices_treinamento]
        
        janela_sel,  indices = k_janela_proxima(x_tr, window_test_inst, k)
        x_janela = x_tr[indices]
        y_saida = y_tr[indices]
        
        modelo = full_ensemble['modelos'][m]
        lag_model = full_ensemble['tam_lags'][m]
        
        erros = []
        for p in range(0,len(y_saida)):
            x_data =  x_janela[p]
            pattern = x_data[0:lag_model+1]
            prev = modelo.predict(pattern)
            err = np.absolute(y_saida[p] - prev[0])
            erros.append(err)
        all_err.append(np.sum(err))
        
    return all_err


def erro_models_val(window_test_inst, x_data_train, y_data_train,  full_ensemble, tam_val, k):

    '''
      erro dos modelos nos k padrões de validação de cada modelo mais próx do novo padrão
    
    '''

    all_err = []
    for m in range(0, len(full_ensemble['modelos'])):
        indices_validacao = full_ensemble['indices'][m][-tam_val:]
        x_val = x_data_train[indices_validacao]
        y_val = y_data_train[indices_validacao]
        
        janela_sel,  indices = k_janela_proxima(x_val, window_test_inst, k)
        x_janela = x_val[indices]
        y_saida = y_val[indices]
        
        modelo = full_ensemble['modelos'][m]
        lag_model = full_ensemble['tam_lags'][m]
        
        erros = []
        for p in range(0,len(y_saida)):
            x_data =  x_janela[p]
            pattern = x_data[0:lag_model+1]
            prev = modelo.predict(pattern)
            err = np.absolute(y_saida[p] - prev[0])
            erros.append(err)
        all_err.append(np.sum(err))
        
    return all_err


def regra_similaridade_dist(window_test_inst,x_data_train, full_ensemble, tam_val, k):    
    #retorna os k janelas de todo treinamento mais próx da janela atual  
    
      
    #verifica a similaridade do treinamento dos modelos com a nova janela
    similaridade = []
    
    
 
    for m in range(0, len(full_ensemble['indices'])):
        
        existe_janelas = []    
        
        indices = full_ensemble['indices'][m][0:tam_val] #todos os indices utilizados para o treinamento do modelo
        x_data_train_model = x_data_train[indices]
        
        k = len(indices)
        
        k_patterns_x, indices, dist = k_janela_proxima_dist(x_data_train_model,  window_test_inst, k)
        
     
        similaridade.append(dist)
    
        
    return similaridade


def select_by_estatistic(window_test_inst, x_data_train, y_data_train, full_ensemble, k):
    
    model_sel = 0
    erro_atual = np.Inf
    
    x_tr = x_data_train
    y_tr = y_data_train
    
    
    estatisticas_train = []
    for i in range(0, len(y_data_train)):
        janela = x_data_train[i,:]
        media = np.mean(janela)
        mediana = np.median(janela)
        desvio = np.std(janela)
        
        estatisticas_train.append([media, mediana, desvio])
    
    
    estatistica_test = [np.mean(window_test_inst), np.median(window_test_inst), np.std(window_test_inst)]
     
    
    #k = 1 #selecionando a estatistica mais próxima!!!!!!
    
    janela_sel,  indices = k_janela_proxima(np.array(estatisticas_train), estatistica_test, k)
    x_janela = x_tr[indices]
    y_saida = y_tr[indices]
    
    erros_modelos = []
    
    
    for m in range(0, len(full_ensemble['modelos'])):       
        
        
        modelo = full_ensemble['modelos'][m]
        lag_model = full_ensemble['tam_lags'][m]
        
        erros = []
        for p in range(0,len(y_saida)):
            x_data =  x_janela[p]
            pattern = x_data[0:lag_model+1]
            prev = modelo.predict(pattern)
            err = np.absolute(y_saida[p] - prev[0])
            erros.append(err)
            
        erros_modelos.append(np.sum(erros))# soma os erros de cada
    
    erro_total = np.sum(erros_modelos)
    tx_err_modelos = np.divide(erros_modelos, erro_total)
        
    return tx_err_modelos 

def selec_model_ola_erro(window_test_inst, past_data, ensemble, k, lags_acf):
    from scipy.spatial.distance import euclidean 

    #retorna o erro dos modelos
   
    #max_lag = len(lags_acf)
    x_data = past_data[:,lags_acf]
    y_data = past_data[:,-1]
    
    

    tam = len(x_data[0]) 
    
    dist = []
    for i in range(0,len(x_data)):
        d = euclidean(window_test_inst, x_data[i,:])
        #print(d)
        dist.append(d)
        
    indices_patterns = range(0, len(x_data))
    
    dist, indices_patterns = zip(*sorted(zip(dist, indices_patterns))) #returna tuplas ordenadas
    #print(indices_patterns)
    indices_patterns_l = list(indices_patterns)

    k_patterns_x = x_data[indices_patterns_l[0:k]]
    k_patterns_y = y_data[indices_patterns_l[0:k]]
    
    
    
    erros_modelos = []
    for i in range(0, len(ensemble['modelos'])):
        model = ensemble['modelos'][i]
        lags = ensemble['tam_lags'][i]
        current_patterns = k_patterns_x[:,0:lags+1]
        
        
        prev = model.predict(current_patterns)
        mse = MSE(k_patterns_y, prev)
        #print('MSE do modelo', i,':', mse)
        erros_modelos.append(mse)       
   
    
    return erros_modelos


def select_model_more(valores_modelos):

    #seleciona o modelo com maior valor 
    
    indices = range(0, len(valores_modelos))
    
    valores_ordenados, indices = zip(*sorted(zip(valores_modelos, indices))) #returna tuplas ordenadas
    
    return indices, indices[-1]


def select_model_less(valores_modelos):

    #seleciona o modelo com menor valor
    
    indices = range(0, len(valores_modelos))
    
    valores_ordenados, indices = zip(*sorted(zip(valores_modelos, indices))) #returna tuplas ordenadas
    
    return indices, indices[0]


def ordenar_desempenho(valores_modelos):
    
    indices = range(0, len(valores_modelos))
    
    valores_ordenados, indices = zip(*sorted(zip(valores_modelos, indices)))
    
    return indices

def pos_sel(selecionado, modelos_ordenados):
    #retorna a posição do modelo selecionado
    pos = 0

    for i in range(0, len(modelos_ordenados)):
        if selecionado == modelos_ordenados[i]:
            pos = i
            break
    return pos


def join_rules_model(values_tr, values_val, values_test, values_sim, values_est, values_ds, coef):
    
    # faz a integração das regras para cada modelo ajustando os coeficientes para ficarem entre 0 e 1
    coef_1 = coef[0] # para regra do desempenho em treinamento
    coef_2 = coef[1] # para regra do desemp em val
    coef_3 = coef[2] # para regra do desemp no test anterior
    coef_4 = coef[3] # para regra da similaridade entre os dados de treinamento do modelo
    coef_5 = coef[4] # para regra da similaridade estatistica
    coef_6 = coef[5] # para regra do DS
	
    
    result = []
    
    for i in range(0, len(values_tr)):
        valor = (coef_1 * values_tr[i]) + (coef_2 * values_val[i]) + (coef_3 * values_test[i]) + (coef_4 * values_sim[i]) + (coef_5 * values_est[i])+ (coef_6 * values_ds[i])
        valor_inverso = np.divide(1,valor)#inverte o valor p se tornar um problema de máximo
        result.append(valor_inverso) 
    
    
    return result


def n_best(prev_ens, target, k):
    
    
    #erro_ens = np.absolute(target - prev_ens)
    erro = []

    for j in range(0, 100): #100 preditores
        e = np.absolute(target - prev_ens[j])
        #print(e)

        erro.append(e) 

    erro_ind_sort = np.argsort(erro) #ordena e retorna os indices
    #print(erro_ind_sort.shape)
    ind_sel = erro_ind_sort[k-1]
    #print('erros:',erro_ind_sort)  
        
        
    return ind_sel



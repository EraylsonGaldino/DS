import numpy as np
from metricas import mse as MSE
def gerar_elm(neuronios, alpha, rbf_w, func_at):
	from sklearn_extensions.extreme_learning_machines.elm import ELMRegressor
	elm = ELMRegressor(n_hidden =neuronios, alpha =alpha, rbf_width = rbf_w, activation_func  = func_at)
	return elm
	
def train_modelo(modelo, x_train, y_train, x_val, y_val, num_ex=5):
	
	melhor_mse = np.Inf
	for i in range(0, num_ex):
		modelo.fit(x_train, y_train)
		prev_v = modelo.predict(x_val)
		novo_mse  = MSE(y_val, prev_v)
		if novo_mse < melhor_mse:
			melhor_mse = novo_mse
			melhor_modelo = modelo
			
	return melhor_modelo, melhor_mse
	
def prev_modelo(modelo, x_test):
	return modelo.predict(x_test)
	
	
def prev_rbf(modelo, x_test):
	return modelo[0].predict(x_test)
	
	
	
	
def gerar_lstm(neuronios,  lags, func_opt='adam'):
	from keras.models import Sequential
	from keras.layers import LSTM, Dense, Dropout
	
	model = Sequential()
	model.add(LSTM(neuronios, input_shape=(1, lags)))
	model.add(Dense(1))
	model.compile(loss='mean_squared_error', optimizer=func_opt)
	
	return model
		
def train_lstm(modelo, x_train, y_train, x_val, y_val, num_ex=5,epochs=100):
	trainX = np.reshape(x_train, (x_train.shape[0], 1, x_train.shape[1]))
	valX = np.reshape(x_val, (x_val.shape[0], 1, x_val.shape[1]))
	melhor_mse = np.Inf
	for i in range(0, num_ex):
		
		modelo.fit(trainX, y_train, epochs=epochs, batch_size=1, verbose=0)
		prev_v = modelo.predict(valX)
		novo_mse  = MSE(y_val, prev_v)
		if novo_mse < melhor_mse:
			melhor_mse = novo_mse
			melhor_modelo = modelo
		
	return melhor_modelo, melhor_mse
	
	
def prev_lstm(modelo, x_test):
	testX = np.reshape(x_test, (x_test.shape[0], 1, x_test.shape[1]))
	y_prev = modelo.predict(testX)
	previsao = []
	for i in range(0, len(y_prev)):
		previsao.append(y_prev[i][0])
	
	
	return previsao
	
def gerar_svr(k, g, e, c):
	from sklearn.svm import SVR
	svr = SVR(kernel=k,gamma=g, epsilon=e, C=c )
	return svr
	
def gerar_mlp(neuronios, func_atv, alg_tr, max_it, tx_apr):
	from sklearn.neural_network import MLPRegressor
	mlp = MLPRegressor(hidden_layer_sizes=neuronios, activation=func_atv, solver=alg_tr, max_iter = max_it, learning_rate= tx_apr)
	
	return mlp



	
def gerar_arvore(max_depth=5):
	from sklearn.tree import  DecisionTreeRegressor
	return DecisionTreeRegressor(max_depth=max_depth)
	
	
def gerar_rbf(qtd_lags, numCent, beta):
	from rbf import RBF 
	rbf = RBF(qtd_lags, numCent, beta)
	return rbf
	

def gerar_esn(n_input, n_reservoir=50, rate=0.2, regrePara=1e-2):
	from easyesn import PredictionESN
	esn = PredictionESN(n_input=n_input, n_output=1, n_reservoir=n_reservoir, leakingRate=rate, regressionParameters=[rate], solver="lsqr", feedback=False)
	return esn
	

	
	
def gerar_dbn(topologia=[10], tx_rbm=0.1, tx_apr=0.3, n_epoc_rbm=0, n_iter=20, funcao='relu'  ):
	from dbn.tensorflow import SupervisedDBNRegression
	
	
	dbn = SupervisedDBNRegression(hidden_layers_structure=topologia,
                                        learning_rate_rbm=tx_rbm,
                                        learning_rate=tx_apr,
                                        n_epochs_rbm=n_epoc_rbm,
                                        n_iter_backprop=n_iter,
                                        batch_size=1,
                                        activation_function=funcao)
										
	return dbn
	
	
def gerar_gcForest(qtd_lags, window=2, tolerance=0.0 ):
	from GCForest import gcForest
	gcf = gcForest(shape_1X=qtd_lags, window=window, tolerance=tolerance)
	return gcf

def gerar_xgboost():
	import xgboost as xgb
	return xgb 


def train_xgb(xgb, params, x_train, y_train, x_val, y_val, num_round):
	dtrain = xgb.DMatrix(x_train, y_train)
	dval = xgb.DMatrix(x_val, y_val)
	bst = xgb.train(params, dtrain, num_round)
	prev_val = bst.predict(dval)
	mse = MSE(y_val, prev_val)
	
	return bst, mse

def prev_xgb(xgb, x_test):
	#utilizar para previsão de apenas 1 ponto
	xTest = np.reshape(x_test, (1, x_test.shape[0]))
	dtest = xgb.DMatrix(xTest)
	prev_test = xgb.predict(dtest)
	return prev_test





def gerar_anf():
	import anfis
	import membership
	return anfis, membership

def train_anf(anfis, membership, mf, x_train, y_train, x_val, y_val, epc=10):
	best_result = np.Inf
	mfc = membership.membershipfunction.MemFuncs(mf)
	anf = anfis.ANFIS(x_train, y_train, mfc)
	anf.trainHybridJangOffLine(epochs=epc)
	prev = anfis.predict(anf, x_val)
	mse  = MSE(y_val, prev)

	return anf, mse

def prev_anf(modelo, x_test):
	import anfis
	prev = anfis.predict(modelo, x_test)
	return prev






def gerar_conv(lags, neuronios=100,filters = 32, kernel_size=2, pool_size= 2, func_opt='adam'):
	from keras.models import Sequential
	from keras.layers import Dense
	from keras.layers import Flatten
	from keras.layers.convolutional import Conv1D
	from keras.layers.convolutional import MaxPooling1D
	model = Sequential()
	model.add(Conv1D(filters=filters, kernel_size=kernel_size, activation='relu',   input_shape=(lags, 1)))
	model.add(MaxPooling1D(pool_size=pool_size))
	model.add(Flatten())
	model.add(Dense(neuronios, activation='relu'))
	
	model.add(Dense(1))
	model.compile(loss='mse', optimizer=func_opt)
		
	return model  
	
	
def gerar_convLSTM(lags, n_lstm=10, n_dense=10, filters=32, kernel_size=2):
	from keras.models import Sequential
	from keras.layers import Dense, Flatten, RepeatVector, LSTM, TimeDistributed
	from keras.layers.convolutional import Conv1D
	from keras.layers.convolutional import MaxPooling1D
	
	model = Sequential()
	model.add(TimeDistributed(Conv1D(filters=filters, kernel_size=kernel_size, activation='relu'), input_shape=(None,lags,1)))
	model.add(TimeDistributed(Conv1D(filters=filters, kernel_size=kernel_size, activation='relu')))
	model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
	model.add(TimeDistributed(Flatten()))
	model.add(LSTM(n_lstm, activation='relu'))
	model.add(Dense(n_dense, activation='relu'))
	model.add(Dense(1))
	model.compile(loss='mse', optimizer='adam')
  
   
	return model
	
def train_convLSTM(modelo, x_train, y_train, x_val, y_val, num_ex=5,epochs=10):
	trainX = x_train.reshape((x_train.shape[0], 1, x_train.shape[1], 1))
	valX = x_val.reshape((x_val.shape[0], 1, x_val.shape[1], 1))
	
	melhor_mse = np.Inf
	for i in range(0, num_ex):
		
		modelo.fit(trainX, y_train, epochs=epochs, batch_size=1, verbose=0)
		prev_v = modelo.predict(valX)
		novo_mse  = MSE(y_val, prev_v)
		if novo_mse < melhor_mse:
			melhor_mse = novo_mse
			melhor_modelo = modelo
		
	return melhor_modelo, melhor_mse 
	
	

def train_esn(modelo, x_train, y_train, x_val, y_val, num_ex=5):
	
	
	melhor_mse = np.Inf
	for i in range(0, num_ex):
		
		modelo.fit(x_train, y_train,transientTime=1, verbose=0)
		prev_v = modelo.predict(x_val)
		novo_mse  = MSE(y_val, prev_v)
		if novo_mse < melhor_mse:
			melhor_mse = novo_mse
			melhor_modelo = modelo
		
	return melhor_modelo, melhor_mse 
	
	
def prev_esn(modelo, x_test):
	previsoes = []
	for i in range(0, x_test.shape[0]):
		
		xTest = np.reshape(x_test[i],(1, (x_test[i].shape[0])))
		prev = modelo.predict(xTest)
		previsoes.append(prev[0][0])
	return previsoes
	
	

	
	
	
def train_conv(modelo, x_train, y_train, x_val, y_val, num_ex=5,epochs=100):
	#[samples, timesteps, rows, cols, channels]
	n_features = 1
	trainX = x_train.reshape((x_train.shape[0], x_train.shape[1], n_features))
	valX = x_val.reshape((x_val.shape[0], x_val.shape[1],n_features))
	
	melhor_mse = np.Inf
	for i in range(0, num_ex):
		
		modelo.fit(trainX, y_train, epochs=epochs, batch_size=1, verbose=0)
		prev_v = modelo.predict(valX)
		novo_mse  = MSE(y_val, prev_v)
		if novo_mse < melhor_mse:
			melhor_mse = novo_mse
			melhor_modelo = modelo
		
	return melhor_modelo, melhor_mse

	
def prev_conv(modelo,x_test ):
	#testX = np.reshape(x_test, (x_test.shape[0], 1, x_test.shape[1]))
	testX = testX.reshape((x_test.shape[0], x_test.shape[1], 1))	
	y_prev = modelo.predict(testX)
	previsao = []
	for i in range(0, len(y_prev)):
		previsao.append(y_prev[i][0]) 
		
	return previsao
	
	
def prev_convLSTM(modelo,x_test ):
	#testX = np.reshape(x_test, (x_test.shape[0], 1, x_test.shape[1]))
	testX = x_test.reshape((x_test.shape[0], 1, x_test.shape[1], 1))
		
	y_prev = modelo.predict(testX)
	previsao = []
	for i in range(0, len(y_prev)):
		previsao.append(y_prev[i][0]) 
		
	return previsao
	

	
	
	
	

	
		
	
	
	
	
		

		


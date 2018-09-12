import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import optimize
'''
Abner Axel López Niño 
A01332362
Red Neuronal para predecir el costo por libra de miel en estados unidos.
Se toma como base para determinar el precio: 
    - Numero de colonias de abejas
    - Libras de miel producidas por colonia
    - Producción total de miel
    - Acciones en manos de los productores (stocks) en la primer quincena de Diciembre del año anterior
    - Año
'''


'''
Definicion de la clase para crear la red neuronal
'''
class RedNeuronal(object):
    def __init__(self, Lambda = 0):
        '''
        inputs: numero de columnas de entrada, las cuales se usarán para determinar
        el valor de la columna de salida en la predicción.
        outputs: numero de columnas de salida. Su valor será el que se prediga a partir
        de las columnas de inputs
        '''
        self.inputs = 5
        self.outputs = 1
        self.hidden = 30 #1,6,7,8,20, 40, 50, 100
        self.W1 = np.random.randn(self.inputs, self.hidden)
        self.W2 = np.random.randn(self.hidden, self.outputs)
        self.Lambda = Lambda
    def sigmoide(self,z):
        return 1/(1+ np.exp(-z))
    def feedForward(self,x):
        self.z2 =  x @ self.W1
        self.a2 =  self.sigmoide(self.z2)
        self.z3 =self.a2 @ self.W2
        self.yhat =  self.sigmoide(self.z3)
        return self.yhat
    def sigmoideDerivada(self, z):
        return np.exp(-z) / ((1 + np.exp(-z)) ** 2)
    def funcionCosto(self,x,y):
        self.yhat = self.feedForward(x)
        Costo = 0.5*sum((y-self.yhat)**2)/x.shape[0] + (self.Lambda/2) * (np.sum(self.W1**2) +np.sum(self.W2**2))
        return Costo
    def funcionDeCostoDerivada(self,x,y):
        self.yhat = self.feedForward(x)
        self.delta3 = np.multiply(-(y - self.yhat ),self.sigmoideDerivada(self.z3))
        djW2 = (np.transpose(self.a2)@self.delta3) / x.shape[0] + (self.Lambda*self.W2)
        self.delta2 = self.delta3@ djW2.T*self.sigmoideDerivada(self.z2)
        djW1 =   (x.T @ self.delta2) / x.shape[0] + (self.Lambda*self.W1)
        return djW1, djW2
    def getPesos(self):
        data = np.concatenate((self.W1.ravel(), self.W2.ravel()))
        return data
    def setPesos(self, datos):
        W1_inicio = 0
        W1_fin = self.hidden * self.inputs
        self.W1 = np.reshape(datos[W1_inicio:W1_fin], (self.inputs, self.hidden))
        W2_fin = W1_fin + self.hidden * self.outputs
        self.W2 = np.reshape(datos[W1_fin:W2_fin], (self.hidden, self.outputs))
    def getGradientes(self, X, y):
        djW1, djW2 = self.funcionDeCostoDerivada(X, y)
        return np.concatenate((djW1.ravel(), djW2.ravel()))
'''
Definicion de la clase entrenador, la cuál procesará una parte del dataset
para entrenar el modelo predictivo
'''
class Entrenador:
    def __init__(self, unaRed):
        # referencia a una red local
        self.NN = unaRed
    def actualizaPesos(self, params):
        self.NN.setPesos(params)
        self.Costos.append(self.NN.funcionCosto(self.X, self.y)) #probar con xtrain y ytrain
        self.CostosTest.append(self.NN.funcionCosto(self.Xtest, self.ytest))
    def obtenPesosNN(self, params, X, y):
        self.NN.setPesos(params)
        cost = self.NN.funcionCosto(X, y)
        grad = self.NN.getGradientes(X, y)
        return cost, grad
    def entrena(self, Xtrain, ytrain, Xtest, ytest):
        '''

        :param Xtrain: Datos de entrada para el entrenamiento del modelo.
        :param ytrain: Datos de salida para el entrenamiento del modelo.
        :param Xtest: Datos de entrada para la predicción a realizar.
        :param ytest: Datos de salida para la predicción a realizar.
        :return:
        '''
        # variables para funciones callback
        self.X = Xtrain
        self.y = ytrain
        self.Xtest = Xtest
        self.ytest = ytest
        # lista temporal de costos
        self.Costos = []
        self.CostosTest = []
        pesos = self.NN.getPesos()
        opciones = {'maxiter': 200, 'disp': True}

        # salida, regresa el costo y los gradientes
        salida = optimize.minimize(self.obtenPesosNN, pesos, jac=True, method='BFGS', args=(Xtrain, ytrain), options=opciones, callback=self.actualizaPesos)
        self.NN.setPesos(salida.x)
        self.resultados = salida
'''
columns: se ingresan los nombres de las columnas que se usarán 
para determinar cierto valor de salida(target).
target: se ingresan los nombres de las columnas de las cuales
se quiere predecir su comportamiento a partir de las columnas
de entrada.
data: se ingresa el nombre del archivo de datos
'''
columns = ['numcol','yieldpercol','totalprod','stocks','year']
target = ['priceperlb']
data = pd.read_csv("honeyproduction.csv")
data.head()
x = data.loc[:, columns]
y = data.loc[:, target]

#Normalización de datos
'''
xtrain y ytrain: 
    Se pone el tamaño de la muestra de entrenamiento. Se utilizó un 80% de los datos.
    Por lo que el total de datos (627) se multiplicó por 0.8 y se ingreso este valor
    más uno (501 + 1). Esto último debido a que la primer fila contiene el nombre de las columnas
xtest y ytest:
    Se pone el resto de los datos. A estos datos se les aplicará la predicción
'''
xtrain = x[1:502]
xtest = x[502:]
ytrain = y[1:502]
ytest = y[502:]
'''
Se procede a normalizar los datos. Se dividen todos los datos entre el valor
máximo de esa columna. Esto con el fin para tener todos los datos con
un valor entre 0 y 1.
'''
xtrain = xtrain/np.amax(xtrain,axis=0)
ytrain = ytrain/np.amax(ytrain,axis=0)
xtest = xtest/np.amax(xtest,axis=0)
ytest = ytest/np.amax(ytest,axis=0)

'''
Se pasan todos los valores a forma de matriz
'''
xtrainMatrix = xtrain.as_matrix()
ytrainMatrix = ytrain.as_matrix()
xtestMatrix = xtest.as_matrix()
ytestMatrix = ytest.as_matrix()

'''
Se crean los objetos red neuronal y entrenador
'''
rn = RedNeuronal(Lambda=0.0001)
e = Entrenador(rn)
'''
Se entrena el modelo y se predice al darle los datos de entrenamiento y predicción en forma de matrices
'''
e.entrena(xtrainMatrix, ytrainMatrix, xtestMatrix, ytestMatrix)

'''
Se grafican los resultados del entrenamiento y de la predicción
'''
plt.plot(e.Costos)
plt.plot(e.CostosTest)
plt.grid(2)
plt.ylabel("costo")
plt.xlabel("iteraciones")
plt.show()


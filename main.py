#  ========Librerias a utilizar============

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#=============Histogramas=================
#Darle titulo
plt.title("Muertos y sobrevivientes")#Histograma de frecuencias
Array=np.array([1,2,3,4,5,6])
#Estructura de parametros (array,bins=5, range=[0,final], ancho, color, borde)
Histograma=plt.hist(Array, bins=6, range=[1,6], rwidth=0.8, color="red", ec="black")
print(Histograma)

#==============Medidas de localizacion y de variabilidad==================
Array=np.array([1,2,3,4,5,6,7,8,9,10])
#Rango
print(Array.ptp())
#media
from statistics import mean
print(mean(Array))
#mediana
from statistics import median
print(median(Array))
#moda
from statistics import mode, multimode
print(mode(Array))
#Para cuando hay varias modas
print(multimode(Array))

#Otra forma de calcular moda
import scipy
from scipy import stats
moda=scipy.stats.mode(Array)
print(moda[0])

#Varianza Poblacional
from numpy import var
varianza=var(Array)
print(varianza)

#Varianza Muestral
from numpy import var
varianza=var(Array, ddof=1)
print(varianza)



#=========PANDAS Y DATAFRAMES===============

#Importar un dataframe
import seaborn as sns
import pandas as pd
df = sns.load_dataset('titanic')

#Mostrar el principio
df.head()

#Mostrar el final
df.tail()

#Saber cuantas filas y columnas
print("filas:",df.shape[0])
print("columnas:",df.shape[1])

#Columnas y atributos
df.info()

#Eliminar Columnas
df.drop('deck', axis=1, inplace=True)

#Eliminar filas con datos incompletos
df.dropna(axis=0, inplace=True)

#Observar de un rango en especifico
df.loc[100:110]

#Colocar nuevos indices
df=df.reset_index()

#Dato maximo y minimo de una columna
df['age'].min()
df['age'].max()

# Mostrar partes especificas
df.query('age==0.42')[['sex','age']]

#Para comparar cuando sucede algo en una fila
df[(df['age']==80)]
#Comparar varias sentencias
df[ (df['survived'] == 1)  &  (df['sex']=='female') & (df['age']>60)] 

#Contar cuantas veces cuando pasa algo
(df[(df['embark_town'] == 'Queenstown')]).count()

#Agrupar por columna en especifico
dfpuerto=df.groupby('embark_town')

#Importar y convertir otro tipo de archivos desde drive
df4=pd.read_csv("/content/drive/MyDrive/UPB/Semestre 2/Procesos Estocasticos/supermarkets.csv")

#Media de una columna
df['age'].mean()

#Saber los cuantiles de una columna
df['age'].quantile([0, .25, .5, .75, 1])

#Desviacion estandar de una columna
df['fare'].std()

#Diagrama de cajas y bigotes de una columna
df["age"].plot.box()

#Diagrama de dispersion entre 2 columnas
df.plot.scatter(x="age",y='fare')


#================Covarianza y Coorelacion=====================


#Crear DataFrame
a=([3,6,9,10,11])
b=([12,22,33,41,46])
df=pd.DataFrame({"a":a,"b":b})


#Coovarianza muestral entre 2 variables a y b
df["a"].cov(df["b"],ddof=1)
#Coovarianza poblacional entre 2 variables a y b
df["a"].cov(df["b"],ddof=0)


#Covarianza de un dataframe
df.cov(ddof=1)





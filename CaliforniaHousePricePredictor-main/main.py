import numpy as np 
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import sklearn.datasets.fetch_california_housing

 #loading dataset
housing=fetch_california_housing()


#spiliting dataset
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test= train_test_split(housing.data,housing,target,random_state=40)


#normalizing dataset
from sklearn.preprocessing import StandardScaler

scaler=StandardScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)

#setting seeds
np.random.seed(40)
tf.random.set_seed(40)

#creation of model

model=keras.models.Sequential([
	#8 diff args
	keras.layers.Dense(40,activation="relu",input_state=[8])
	keras.layers.Dense(40,activation="relu")
	#regression model one singe output
	keras.layers.Dense(1)

	])
#compile
model.compile(loss="mean_squared_error",optimizer=keras.optimizers.SGD(lr=1e-3),mertics=['mae'])


#fitting of data

done_model=model.fit(x_train,y_train,epochs=80)


#prediction

pred=done_model.predict(x_train)
print(pred)
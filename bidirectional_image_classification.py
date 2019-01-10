from keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import roc_auc_score

(x_train,y_train),(x_test,y_test)=mnist.load_data()
y_train=np.reshape(y_train,(len(y_train),1))
y_test=np.reshape(y_test,(len(y_test),1))
labels=OneHotEncoder()
y_train=labels.fit_transform(y_train)
y_test=labels.fit_transform(y_test)


from keras.layers import Dense,Input,GlobalMaxPool1D,GRU,Bidirectional,Lambda,Concatenate
from keras.optimizers import Adam
from keras.models import Model
import keras.backend as K
from keras.layers import Dropout

input_=Input(shape=(28,28))
rnn1=Bidirectional(GRU(20,activation='sigmoid',return_sequences=True))
x1=rnn1(input_)
x1=GlobalMaxPool1D()(x1)

permutor=Lambda(lambda t:K.permute_dimensions(t,pattern=(0,2,1)))

rnn2=Bidirectional(GRU(20,activation='sigmoid',return_sequences=True))
x2=permutor(input_)
x2=rnn2(x2)
x2=GlobalMaxPool1D()(x2)

concatenator=Concatenate(axis=1)
x=concatenator([x1,x2])
x=Dense(128,activation='sigmoid')(x)

output=Dense(10,activation='sigmoid')(x)
model=Model(input_,output)
model.compile(loss='categorical_crossentropy',metrics=['accuracy'],optimizer=Adam(lr=0.01))
his=model.fit(x_train,y_train,epochs=100,validation_split=0.2,batch_size=128)

plt.plot(his.history['loss'],label='loss')
plt.plot(his.history['val_loss'],label='validation_loss')
plt.legend()
plt.show()

plt.plot(his.history['acc'],label='accuracy')
plt.plot(his.history['val_acc'],label='val_accuracy')
plt.legend()
plt.show()

pre=model.predict(x_test)

print(np.mean(abs(pre-y_test)))


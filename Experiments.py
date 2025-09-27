import datetime
from tensorboard import program
import tensorboard
import pandas as pd
from PIL.ImageOps import scale
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,LabelEncoder
import pickle
from sklearn.preprocessing import OneHotEncoder
import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping,TensorBoard

data=pd.read_csv("Churn_Modelling.csv")
data.head(8)
#data preprocessing
data=data.drop(['RowNumber','CustomerId','Surname'],axis=1)
#encode categorical values
label_encoder_gender=LabelEncoder()
data['Gender']=label_encoder_gender.fit_transform(data['Gender'])
#print(data)

##One hot encoding for Geography column
one_hot_encoding_geography=OneHotEncoder()
geo_encoder=one_hot_encoding_geography.fit_transform(data[['Geography']])
data_frame=pd.DataFrame(geo_encoder.toarray(),columns=one_hot_encoding_geography.get_feature_names_out(['Geography']))
#combining one hot encoders with the original data
combining_data=pd.concat([data.drop(['Geography'],axis=1),data_frame],axis=1)
combining_data.head(3)
#print(combining_data)
#save the encoders and scaler
with open('label_encoder_gender.pkl','wb') as file:
    pickle.dump(label_encoder_gender,file)

with open('geo_encoder.pkl','wb') as file:
    pickle.dump(one_hot_encoding_geography,file)

X=combining_data.drop('Exited',axis=1)
Y=data['Exited']

#Split the data into training and testing sets
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=42)

scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.fit_transform(X_test)
'''print(X_train)
print(X_test)'''
#Creating a pickle file for scaler
with open('scaler.pkl','wb') as file:
    pickle.dump(scaler,file)

model=Sequential([
    Dense(64,activation='relu', input_shape=(X_train.shape[1],)), #Hidden layer 1 connected with Input layer
    Dense(32,activation='relu'), #Hidden layer 2
    Dense(1,activation='sigmoid') #Output layer
])

'''model.summary()'''

opt=tensorflow.keras.optimizers.Adam(learning_rate=0.01)

model.compile(optimizer=opt,loss="binary_crossentropy",metrics=['accuracy'])

#Setup the tensorboard

log_dir="log/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorflow_callback=TensorBoard(log_dir=log_dir,histogram_freq=1)

#setup early stopping
early_stopping_callback=EarlyStopping(monitor='val_loss', patience=10,restore_best_weights=True)

history=model.fit(
    X_train,Y_train,validation_data=(X_test,Y_test), epochs=100, callbacks=[tensorflow_callback,early_stopping_callback]
)

'''print(history)'''
#save H5 file
model.save('model.h5')

tb = program.TensorBoard()
tb.configure(argv=[None, f'--logdir={log_dir}'])
url = tb.launch()
print(f"TensorBoard is running at {url}")
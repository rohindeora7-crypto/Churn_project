import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle
import pandas as pd
import numpy as np

#load the scaled model
model=load_model('model.h5')

#opening of pickle files
with open('geo_encoder.pkl','rb') as file:
    geo_encoder=pickle.load(file)

with open('label_encoder_gender.pkl','rb') as file:
    label_gender=pickle.load(file)

with open('scaler.pkl','rb') as file:
    scaler=pickle.load(file)

#input data
input_data = {
    'CreditScore': 600,
    'Geography': 'France',
    'Gender': 'Male',
    'Age': 40,
    'Tenure': 3,
    'Balance': 60000,
    'NumOfProducts': 2,
    'HasCrCard': 1,
    'IsActiveMember': 1,
    'EstimatedSalary': 50000
}
#OHE used to convert feature representation
geo_encoded = geo_encoder.transform([[input_data['Geography']]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=geo_encoder.get_feature_names_out(['Geography']))

input_df=pd.DataFrame([input_data])
#Encode categorical variables
input_df['Gender']=label_gender.transform(input_df['Gender'])
#OHE converting words to vector
input_df=pd.concat([input_df.drop('Geography', axis=1),geo_encoded_df],axis=1)
#print(input_df)

## Scaling the input data
scaled=scaler.transform(input_df)
#print(scaled)

#predicting the churn
prediction=model.predict(scaled)
#print(prediction)

prediction_prob=prediction[0][0]
print(prediction_prob)

if prediction_prob>0.5:
    print("The customer will churn")

else:
    print("The customer is not likely to churn")




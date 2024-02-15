import pickle as pkl
import sklearn
import numpy as np

log_model = pkl.load(open('log_model.pkl', 'rb'))
heart_data = pkl.load(open('heart_data.pkl', 'rb'))

#uncomment the below and then assign the variables their values from either json script or flask module or 
#wherever you have stored the value

age = 25
sex = 1
cp =  4
trestbps = 0
chol = 0
fbs = 1
restecg = 0
thalach = 0
exang = 0
oldpeak = 0.1 
slope = 2
ca = 1
thal = 1 

query = np.array([age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal])
query = query.reshape(1, 13)
ans = log_model.predict(query)
print(ans)


#ans will be either 0 or 1 where 0 = no heart attack possibility and 1 = heart attack possibility  
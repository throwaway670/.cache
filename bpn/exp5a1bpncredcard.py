import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report,accuracy_score
from imblearn.over_sampling import SMOTE

data=pd.read_csv('creditcard.csv')
X=data.drop('Class',axis=1).values
y=data['Class'].values
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42,stratify=y) # type: ignore
scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)
sm=SMOTE(random_state=42)
X_train_res,y_train_res=sm.fit_resample(X_train,y_train) # type: ignore

def sigmoid(x):
    return 1/(1+np.exp(-x))
def sigmoid_derivative(x):
    return x*(1-x)
input_dim=X_train_res.shape[1]
hidden_dim=16
output_dim=1
lr=0.01
epochs=100
np.random.seed(42)
W1=np.random.uniform(size=(input_dim,hidden_dim))
b1=np.zeros(hidden_dim)
W2=np.random.uniform(size=(hidden_dim,output_dim))
b2=np.zeros(output_dim)
for epoch in range(epochs):
    z1=np.dot(X_train_res,W1)+b1
    a1=sigmoid(z1)
    z2=np.dot(a1,W2)+b2
    a2=sigmoid(z2)
    error=y_train_res.reshape(-1,1)-a2 # type: ignore
    d_a2=error*sigmoid_derivative(a2)
    d_a1=d_a2.dot(W2.T)*sigmoid_derivative(a1)
    W2+=a1.T.dot(d_a2)*lr
    b2+=np.sum(d_a2,axis=0)*lr
    W1+=X_train_res.T.dot(d_a1)*lr
    b1+=np.sum(d_a1,axis=0)*lr
    if (epoch+1)%10==0:
        loss=np.mean(np.abs(error))
        print(f"Epoch {epoch+1}/{epochs},Loss:{loss:.4f}")
z1_test=np.dot(X_test,W1)+b1
a1_test=sigmoid(z1_test)
z2_test=np.dot(a1_test,W2)+b2
a2_test=sigmoid(z2_test)
y_pred=(a2_test>0.5).astype(int).flatten()
print("\nTest Accuracy:",accuracy_score(y_test,y_pred))
print("\nClassification Report:\n",classification_report(y_test,y_pred,digits=4))

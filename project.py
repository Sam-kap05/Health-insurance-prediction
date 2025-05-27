import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv('insurance.csv')

df.info()

df.head()

def binary_map(x):
    return x.map({'yes':1,'no':0})

def gender_map(x):
    return x.map({'female':1,'male':0})

def encdummy(df,name):
    dummies = pd.get_dummies(df[name])
    for x in dummies.columns:
        dname = f"{name}-{x}"
        df[dname] = dummies[x]
    df.drop(name,axis=1,inplace = True)

def encmap(x): 
    return x.map({True:1, False:0})

col1 = ['smoker']
col2 = ['region-southwest','region-southeast','region-northwest','region-northeast']
col3 = ['sex']
df[col1] = df[col1].apply(binary_map)
encdummy(df,'region')
df[col2] = df[col2].apply(encmap)
df[col3] = df[col3].apply(gender_map)

from sklearn.preprocessing import StandardScaler
X = df.drop('charges',axis = 1)
y = df['charges']
import numpy as np

class stdscaler:
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, X):
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        return self
    
    def transform(self, X):
        return (X - self.mean) / self.std
    
    def fittransform(self, X):
        self.fit(X)
        return self.transform(X)

scaler = stdscaler()
X_scaled = scaler.fittransform(X)
ymean = np.mean(y)
ystd = np.std(y)
y = scaler.fittransform(y)
X_train=X_scaled

df.head()

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

X_train, X_test, y_train, y_test = train_test_split(X_train, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train,y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R-squared:", r2)

# Scatter plot: Actual vs Predicted
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.6, label="Predicted vs Actual")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', label="y = x")
plt.xlabel("Actual Charges")
plt.ylabel("Predicted Charges")
plt.title("Actual vs Predicted Medical Charges")
plt.legend()
plt.grid(True)
plt.show()

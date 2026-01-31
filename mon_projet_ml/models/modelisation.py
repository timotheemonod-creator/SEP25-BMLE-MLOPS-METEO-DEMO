import numpy as np
import pandas as pd                           
import warnings
warnings.filterwarnings("ignore")
 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier 

import joblib

import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv("../data/raw/weatherAUS.csv")

#supprimer NA de la variable cible
df = df.dropna(subset=['RainTomorrow'])

#Forcer les outliers en nan
df.loc[df['WindSpeed9am'] > 100, 'WindSpeed9am'] = np.nan
df.loc[df['Evaporation'] > 140, 'Evaporation'] = np.nan
df.loc[df['Rainfall'] > 350, 'Rainfall'] = np.nan

# Convert Date column to datetime and extract day, month
df['Date'] = pd.to_datetime(df['Date'])
df['Dayofyear'] = df['Date'].dt.dayofyear

df=df.drop(columns=['Date'])

df=df.drop(columns=['Temp3pm','Temp9am'])

X = df.drop("RainTomorrow", axis=1)
y = df["RainTomorrow"]


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)

y_train = (y_train == "Yes").astype(int)
y_test  = (y_test  == "Yes").astype(int)

cat_cols = X_train.select_dtypes(include=["object"]).columns


# Filling missing values with mode of the column in value
for col in cat_cols:
    X_train[col] = X_train[col].fillna(X_train[col].mode()[0])
    X_test[col] = X_test[col].fillna(X_test[col].mode()[0])


num_cols = X_train.select_dtypes(include=["int32", "float64"]).columns
print (f"the numerical variable is \n {num_cols}" )

# Filling missing values with median of the column in value
for col in num_cols:
    X_train[col] = X_train[col].fillna(X_train[col].mean())
    X_test[col] = X_test[col].fillna(X_test[col].mean())

def sin_transform(x, period):
    return np.sin(x / period * 2 * np.pi)

def cos_transform(x, period):
    return np.cos(x / period * 2 * np.pi)

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore",sparse_output=False), cat_cols),
        ("dayofyear_sin", FunctionTransformer(sin_transform, kw_args={"period": 365}), ["Dayofyear"]),
        ("dayofyear_cos", FunctionTransformer(cos_transform, kw_args={"period": 365}), ["Dayofyear"]),
    ],
    remainder="passthrough"
)
pipeline = Pipeline(
    steps=[
        ("preprocessing", preprocessor),
        ("scaling", StandardScaler()),
        ("smote", SMOTE(random_state=42)),
        ("model", XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=42
        ))
    ]
)

pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)
y_proba = pipeline.predict_proba(X_test)[:, 1]


joblib.dump(pipeline, '../models/pipeline.pkl')


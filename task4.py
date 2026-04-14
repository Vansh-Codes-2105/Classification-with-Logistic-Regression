import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_score, recall_score, roc_auc_score


df = pd.read_csv("data.csv")  

if 'id' in df.columns:
    df = df.drop('id', axis=1)


if 'diagnosis' in df.columns:
    df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})


X = df.drop('diagnosis', axis=1)
y = df['diagnosis']


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)


y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]


print("\n--- MODEL EVALUATION ---")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nPrecision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_prob))


threshold = 0.3

y_pred_new = (y_prob >= threshold).astype(int)

print("\n--- AFTER THRESHOLD TUNING (0.3) ---")
print(confusion_matrix(y_test, y_pred_new))


sample = X_test[0].reshape(1, -1)
prediction = model.predict(sample)

print("\nSample Prediction (0 = Benign, 1 = Malignant):", prediction[0])
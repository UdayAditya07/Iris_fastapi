import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,classification_report

model = LogisticRegression(max_iter=500)
df = pd.read_csv('Iris.csv')
df= df.drop('Id',axis=1)
X = df[["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]]
y = df["Species"]


X_train, X_test, y_train, y_test = train_test_split(
    X,y, test_size=0.2, random_state=42
)

model.fit(X_train,y_train)

y_pred = model.predict(X_test)
# species_map = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}
# y_numeric = y.map(species_map)

# # Evaluate
# accuracy = accuracy_score(y_test, y_pred)
# print("Accuracy:", accuracy)
# print("Classification Report:\n", classification_report(y_test, y_pred))

# # Optional: Plotting the features
# plt.scatter(X['PetalLengthCm'], X['PetalWidthCm'], c=y_numeric, cmap='viridis', edgecolor='k')
# plt.xlabel('Petal Length (cm)')
# plt.ylabel('Petal Width (cm)')
# plt.title('Iris Dataset Visualization')
# plt.show()

joblib.dump(model, "model.joblib")
joblib.dump(y, "target_names.joblib")
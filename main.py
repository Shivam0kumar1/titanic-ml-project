import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score

# Load the real Titanic data
df = pd.read_csv("train.csv")

# Keep only columns we need and drop rows without data
df = df[['Survived', 'Pclass', 'Sex', 'Age']].dropna()
df["Sex"] = df["Sex"].map({'male':0, 'female':1})

print(df.head())    # Shows top 5 rows to see what the data looks like

# Features and Target
X = df[['Pclass', 'Sex', 'Age']]
y = df['Survived']

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X,y,train_size=0.3, random_state=0)

# Train the model with train sets
model = LinearRegression()
model.fit(X_train, y_train)
print("Model training complete!")

# Check accuracy
predictions = model.predict(X_test)
if predictions.ndim == 1 and predictions.dtype!=int:
    predictions: bytes = (predictions >= 0.5).astype(int)
print("Accuracy: ", accuracy_score(y_test, predictions))
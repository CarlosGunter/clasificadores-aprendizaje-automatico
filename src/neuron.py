import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import sklearn.neural_network as nn
from imblearn.over_sampling import SMOTE

# Seleccionar durante entrenamiento
# Duplicar entradas desbalanceadas
# Usar SMOTE

mapActivities = {
    1: 'WALKING',
    3: 'SHUFFLING',
    4: 'UPSTAIRS',
    5: 'DOWNSTAIRS',
    6: 'STANDING',
    7: 'SITTING',
    8: 'LAYING'
}
mapInputs = {
    1: 0,
    3: 1,
    4: 2,
    5: 3,
    6: 4,
    7: 5,
    8: 6
}
mapOutputs = {
    0: 1,
    1: 3,
    2: 4,
    3: 5,
    4: 6,
    5: 7,
    6: 8
}
# Def target output
def targetOutput(activity):
    i = mapInputs[activity]
    target = [0] * 7
    target[i] = 1.0
    return target

# Get data
def getData(path):
    data = np.genfromtxt(f"assets/{path}.csv", dtype=float, delimiter=',', skip_header=1, usecols=range(1, 8))
    return data

# Load data
dataI = getData(501)
for i in range(502, 513):
    dataI = np.vstack((dataI, getData(i)))
data, classes = dataI[:,:-1], dataI[:,-1]
print(f"Data shape: {data.shape}")
print(f"Data shape: {classes.shape}")
print("Data loaded")

# Normalize data
scaler = StandardScaler()
data = scaler.fit_transform(data)
print("\nData normalized")

data, classes = SMOTE().fit_resample(data, classes)
print(f"Data shape: {len(data)}")
print("Data balanced")


# Split data
data, data_test, classes, classes_test = train_test_split(data, classes, test_size=0.2, random_state=0)
print("Data splitted")
print(f"Data shape: {len(data)}")

# Initialize network
model = nn.MLPClassifier(
    activation='tanh',
    hidden_layer_sizes=(28, 28, 42, 42, 42, 42, 28, 28,),
    verbose=True,
    max_iter=300,
    tol=1e-5,
)
# Train network
model.fit(data, classes)
# Predict
print("-----------------------------------")
predictions = model.predict(data_test)
print(f"Accuracy: {accuracy_score(classes_test, predictions)*100}%")
print(classification_report(classes_test, predictions))

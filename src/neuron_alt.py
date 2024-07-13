import lib.neuron as nrn
import numpy as np

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

# Initialize network (from neuron.py)
inSz, hddnSz, outSz, initWSD, lrnRate= 6, 24, 7, 0.25, 0.75
global net
net = nrn.NN(inSz, hddnSz, outSz, initWSD, lrnRate)
print('-----------------------------------')
print(f'Network with ({inSz}, {hddnSz}, {outSz}, {initWSD}, {lrnRate})')
# Train network
trainings = 5
numbClasses = [0] * 7
for i in range(trainings):
    print('-----------------------------------')
    print(f'Training {i+1}/{trainings}')
    # Data training
    for j in range(len(data)):
        # Calculate classes
        if i == 0:
            numbClasses[mapInputs[classes[j]]] += 1
        # Calculate output
        actualOutput,hiddenOutput = net.calcNetOutput(
            data[j],
            wantHiddenLevels=True
        )
        # Training episode
        net.trainingEpisode(
            targetOutput(classes[j]),
            actualOutput,
            hiddenOutput,
            data[j]
        )

# Define activity output
def activityOutput(output):
    return mapOutputs[output.index(max(output))]

# Test network
def test():
    print('-----------------------------------')
    print('Testing')
    testData = np.genfromtxt('assets/507.csv', dtype=float, delimiter=',', skip_header=1, usecols=range(1, 8))
    testData, testClasses = testData[:,:-1], testData[:,-1]
    # testData = scaler.fit_transform(testData)
    trueActivities = np.zeros(7)
    activities = np.zeros(7)
    corrects = 0
    for i in range(len(testData)):
        output = net.calcNetOutput(testData[i])
        if testClasses[i] == activityOutput(output):
            corrects += 1
            trueActivities[mapInputs[testClasses[i]]] += 1
        activities[mapInputs[testClasses[i]]] += 1
    return corrects, len(testData), trueActivities, activities

corrects, total, trueActivities, activities = test()
print(f'Corrects: {corrects/total*100}%')
print(f'{sum(trueActivities)} corrects of {total} activities')
for i in range(len(activities)):
    print(f'{mapActivities[mapOutputs[i]]} corrects: {trueActivities[i]}/{activities[i]}')
print('-----------------------------------')
print(numbClasses)
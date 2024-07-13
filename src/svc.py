import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


atributos = np.genfromtxt('assets/TUANDROMD.csv', delimiter=',', dtype=int, encoding=None, skip_header=1, usecols=range(0, 241))
tipo = np.genfromtxt('TUANDROMD.csv', delimiter=',', dtype=str, encoding=None, skip_header=1, usecols=-1)
tipo_int = np.array([1 if x == 'malware' else 0 for x in tipo])

X_train, X_test, y_train, y_test = train_test_split(atributos, tipo, test_size=0.2, random_state=0)

def SVC():
    from sklearn.svm import SVC

    clf = SVC(gamma='auto')
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    print(classification_report(y_test, y_pred, zero_division=0))
    print(f"Accuracy: {accuracy_score(y_test, y_pred, normalize=True)*100}%")
    print("--------------------")
    print(f"Confusion matrix:\n")
    print(confusion_matrix(y_test, y_pred))

SVC()
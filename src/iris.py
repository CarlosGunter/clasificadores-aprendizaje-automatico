import numpy as np
import lib.neuron as neuron

datos = np.genfromtxt("../assets/iris.csv",dtype=float,delimiter=",")

setosa = datos[0:50,:]
versicolor = datos[50:100,:]
virginica = datos[100:150,:]

print(setosa[0])
print(setosa[0,-1])

red = neuron.NN(4,3,3,0.5,0.7)

numFlorEntrena = 30
entrenamientos = 3000

for i in range(entrenamientos):
    for j in range(numFlorEntrena):
        actualOutput,hiddenOutput = red.calcNetOutput(setosa[j],wantHiddenLevels=True)
        red.trainingEpisode([1,0,0], actualOutput, hiddenOutput, setosa[j])
        actualOutput,hiddenOutput = red.calcNetOutput(versicolor[j],wantHiddenLevels=True)
        red.trainingEpisode([0,1,0], actualOutput, hiddenOutput, versicolor[j])
    
        actualOutput,hiddenOutput = red.calcNetOutput(virginica[j],wantHiddenLevels=True)
        red.trainingEpisode([0,0,1], actualOutput, hiddenOutput, virginica[j])
    
def prueba(red,datos):
    cuentas=[0,0,0]
    for i in range(150):
        salida=red.calcNetOutput(datos[i])
        if(i<50):
            if(salida[0]>0.5 and salida[1]<0.5 and salida[2]<0.5):
                cuentas[0]+=1
        elif(i<100):
            if(salida[1]>0.5 and salida[0]<0.5 and salida[2]<0.5):
                cuentas[1]+=1
        else:
            if(salida[2]>0.5 and salida[0]<0.5 and salida[1]<0.5):
                cuentas[2]+=1
    print("Setosas correctas: ",cuentas[0]/50*100,"%")
    print("Versicolor correctas: ",cuentas[1]/50*100,"%")
    print("Virginica correctas: ",cuentas[2]/50*100,"%")
    print("Flores correctas: ",sum(cuentas)/150*100,"%")
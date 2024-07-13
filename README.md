# Clasificadores - Aprendizaje Automático
Se evalúan distintos clasificadores de Aprendizaje Automático con scripts de python donde se incluye preprocesamiento de datos.
Para iniciar los scripts es necesario descargar los datasets disponibles en el repositorio de [UC Irvine](https://archive.ics.uci.edu/) y colocar los archivos .csv en el directorio assets.
- [Iris](https://archive.ics.uci.edu/dataset/53/iris)
- [TUANDROMD](https://archive.ics.uci.edu/dataset/855/tuandromd+(tezpur+university+android+malware+dataset))
- [HAR70+](https://archive.ics.uci.edu/dataset/780/har70)

Instalar las dependencias indicadas en los [requerimientos](requeriments.txt)
~~~
$ pip install -r requirements.txt
~~~
## Clasificadores
- Maquinas de vectores de soporte en [TUANDROMD](https://archive.ics.uci.edu/dataset/855/tuandromd+(tezpur+university+android+malware+dataset)) ([Ver](svc.py))
- Red Neuronal en [Iris](https://archive.ics.uci.edu/dataset/53/iris) ([Ver](iris.py))
- Red neuronal en [HAR70+](https://archive.ics.uci.edu/dataset/780/har70) ([Ver](neuron.py))
- Red neuronal en [HAR70+](https://archive.ics.uci.edu/dataset/780/har70) y librería independiente ([Ver](neuron_alt.py))
## Preprocesamiento
- PCA
- Split
- Normalización de datos
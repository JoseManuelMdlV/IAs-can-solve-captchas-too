# IAs-can-solve-captchas-too

## Introducción
<p align="justify"> Las contraseñas fuertes son necesarias para evitar intentos de hackeo mediante ataques de fuerza bruta o ataques de diccionario, pero no es un método infalible. Con la creciente capacidad de los ordenadores, el crackeo de contraseñas se va volviendo cada vez más sencillo y la creación de nuevas técnicas, así como la venida de nuevas tecnologías, como los ordenadores cuánticos, hacen inviable incluso el cifrado de estas</p>

<p align="justify">Para parar los intentos reiterados de un programa por descubrir una contraseña, se encuentra el bloqueo de cuentas (que solo retrasa lo inevitable), y la resolución de un problema que solo un ser humano, a priori, sería capaz de resolver.</p>

<p align="justify">Uno de los métodos más populares para confirmar si el usuario que intenta acceder al dispositivo es un humano o un bot ejecutando un script es el de resolver un captcha, leer los caracteres en una imagen y responder a la pregunta introduciendo como texto dichos caracteres.</p>

<p align="justify">Sin embargo, con el avance en IAs capaces de hacer un reconocimiento de la imagen y convertirla a texto empleando redes neuronales, este método se está volviendo poco seguro y se necesita encontrar nuevas formas de proteger nuestros datos o de crear retos de difícil resolución.</p>

<p align="justify">En este proyecto, que apunta a crear una API capaz de mejorar la seguridad de los sistemas informáticos mediante una autentificación más potente, vamos a crear un modelo de Red Neuronal capaz de reconocer imágenes y leer su contenido para luego resolver un problema que solo un ser humano sería capaz de resolver, quedando a futuro la intención de que este modelo pudiese sugerir captchas que no fuese capaz de resolver, de manera que las herramientas de los delincuentes no pudiesen penetrar esta barrera de seguridad. </p>

## Pasos previos a la creación del modelo

<p align="justify">Comenzamos importando las librerías que nos van a hacer falta para el desarrollo de nuestro modelo, así como cargamos los datos que usaremos para entrenarlo y hacer tests de validación.</p>

<p align="justify">Los datos que se emplean en este proyecto pertenecen al dataset propiedad de FOURNIERP</p>

```python
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns

from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

img_folder = '[PATH TO FOLDER]'

img_2g7nm = mpimg.imread(img_folder + '2g7nm.png')
img_34pcn = mpimg.imread(img_folder + '34pcn.png')
img_bny23 = mpimg.imread(img_folder + 'bny23.png')
img_c4mcm = mpimg.imread(img_folder + 'c4mcm.png')
img_3c7de = mpimg.imread(img_folder + '3c7de.jpg')
img_nxf2c = mpimg.imread(img_folder + 'nxf2c.jpg')
img_pcmcc = mpimg.imread(img_folder + 'pcmcc.jpg')
img_yge7c = mpimg.imread(img_folder + 'yge7c.jpg')
samples = {'2g7nm.png':img_2g7nm, '34pcn.png':img_34pcn, 'bny23.png':img_bny23, 'c4mcm.png':img_c4mcm,
           '3c7de.jpg':img_3c7de, 'nxf2c.jpg':img_nxf2c, 'pcmcc.jpg':img_pcmcc, 'yge7c.jpg':img_yge7c}

fig=plt.figure(figsize=(20, 5))
pos = 1
for filename, img in samples.items():
    fig.add_subplot(2, 4, pos)
    pos = pos+1
    plt.imshow(img)
    plt.title('filename='+filename+' shape='+str(img.shape))
plt.show()
```
<p align="justify">De esta manera, podemos comprobar que hemos cargados los datos correctos, los cuales son imágenes de captchas que podríamos encontrar en cualquier portal de logueo.</p>

<img width="1604" height="360" alt="image" src="https://github.com/user-attachments/assets/ec91a418-48f9-4a75-ab0d-ba4cb8d0385b" />

## Métrica de evaluación

```python
def compute_perf_metric(predictions, groundtruth):
    if predictions.shape == groundtruth.shape:
        return np.sum(predictions == groundtruth)/(predictions.shape[0]*predictions.shape[1])
    else:
        raise Exception('Error : the size of the arrays do not match. Cannot compute the performance metric')
```

<p align="justify">La función definida en la celda será la que emplearemos para poder calcular la precisión del modelo. La idea es poder tomar cada uno de los vectores (5,1) de los que se compondrán tanto las etiquetas como los resultados de las predicciones y se comprobarán posición por posición.</p>

<p align="justify">Un ejemplo: Supongamos que el captcha tiene como etiqueta 5nm2q, y la predicción nos dice que el captcha es 5pm2w.</p>

<p align="justify">Si llamamos Pi a la posición del caracter dentro del vector de tal manera que P1 será la primera posición, la (1,1), P2 la segunda y así hasta P5, que sería la última, nuestra función compara las parejas Pi entre sí. Si ambas Pi coinciden, el valor será un 1; si no lo hacen, un 0.</p>

<p align="justify">Tras comprobar todas las posiciones, se suman los resultados y se divide por el número de posiciones, dando lugar a un resultado que será el porcentaje de acierto del modelo.</p>

<p align="justify">Todo esto será cierto siempre y cuando los vectores de la etiqueta y la predicción posean el mismo tamaño. Si no fuese así, la función daría error.</p>

## Creación de los sets de entrenamiento y validación

```Python
# Dictionaries that will be used to convert characters to integers and vice-versa
vocabulary = {'2','3','4','5','6','7','8','b','c','d','e','f','g','m','n','p','w','x','y'}
char_to_num = {'2':0,'3':1,'4':2,'5':3,'6':4,'7':5,'8':6,'b':7,'c':8,'d':9,'e':10,'f':11,'g':12,'m':13,'n':14,'p':15,'w':16,'x':17,'y':18}

##############################################################################################################################
# This function encodes a single sample.
# Inputs :
# - img_path : the string representing the image path e.g. '/kaggle/input/captcha-version-2-images/samples/samples/6n6gg.jpg'
# - label : the string representing the label e.g. '6n6gg'
# - crop : boolean, if True the image is cropped around the characters and resized to the original size.
# Outputs :
# - a multi-dimensional array reprensenting the image. Its shape is (50, 200, 1)
# - an array of integers representing the label after encoding the characters to integer. E.g [6,16,6,14,14] for '6n6gg'
##############################################################################################################################
def encode_single_sample(img_path, label, crop):
    # Read image file and returns a tensor with dtype=string
    img = tf.io.read_file(img_path)
    # Decode and convert to grayscale (this conversion does not cause any information lost and reduces the size of the tensor)
    # This decode function returns a tensor with dtype=uint8
    img = tf.io.decode_png(img, channels=1)
    # Scales and returns a tensor with dtype=float32
    img = tf.image.convert_image_dtype(img, tf.float32)
    # Crop and resize to the original size :
    # top-left corner = offset_height, offset_width in image = 0, 25
    # lower-right corner is at offset_height + target_height, offset_width + target_width = 50, 150
    if(crop==True):
        img = tf.image.crop_to_bounding_box(img, offset_height=0, offset_width=25, target_height=50, target_width=125)
        img = tf.image.resize(img,size=[50,200],method='bilinear', preserve_aspect_ratio=False,antialias=False, name=None)
    # Transpose the image because we want the time dimension to correspond to the width of the image.
    img = tf.transpose(img, perm=[1, 0, 2])
    # Converts the string label into an array with 5 integers. E.g. '6n6gg' is converted into [6,16,6,14,14]
    label = list(map(lambda x:char_to_num[x], label))
    return img.numpy(), label

def create_train_and_validation_datasets(crop=False):
    # Loop on all the files to create X whose shape is (1040, 50, 200, 1) and y whose shape is (1040, 5)
    X, y = [],[]

    for _, _, files in os.walk(img_folder):
        for f in files:
            # To start, let's ignore the jpg images
            label = f.split('.')[0]
            extension = f.split('.')[1]
            if extension=='png':
                img, label = encode_single_sample(img_folder+f, label,crop)
                X.append(img)
                y.append(label)

    X = np.array(X)
    y = np.array(y)

    # Split X, y to get X_train, y_train, X_val, y_val
    X_train, X_val, y_train, y_val = train_test_split(X.reshape(1040, 10000), y, test_size=0.1, shuffle=True, random_state=42)
    X_train, X_val = X_train.reshape(936,200,50,1), X_val.reshape(104,200,50,1)
    return X_train, X_val, y_train, y_val
```
<p align="justify">Comenzamos primero creando un diccionario que nos permitirá hacer una traducción de los caracteres a números enteros. Esto se debe realizar así en tanto que inicialmente los caracteres serán vistos como datos de tipo string y la red neuronal no será capaz de comprenderlos (solo entiende de números), no pudiendo realizar operaciones sobre ellos.</p>

<p align="justify">Posteriormente, crearemos funciones con las cuales nos aseguraremos de que todos los tensores tendrán las mismas dimensiones. También haremos la conversión de las etiquetas de strings a números enteros y separaremos el dataset en los datos de entrenamiento (90%) y los de validación (10%).</p>

```python
X_train, X_val, y_train, y_val = create_train_and_validation_datasets(crop=True)
X_train_, X_val_, y_train_, y_val_ = create_train_and_validation_datasets(crop=False)

fig=plt.figure(figsize=(20, 10))
fig.add_subplot(2, 4, 1)
plt.imshow(X_train[0], cmap='gray')
#plt.imshow(X_train[0].transpose((1,0,2)), cmap='gray')
plt.title('Image from X_train with label '+ str(y_train[0]))
plt.axis('off')
fig.add_subplot(2, 4, 2)
plt.imshow(X_train[935], cmap='gray')
#plt.imshow(X_train[935].transpose((1,0,2)), cmap='gray')
plt.title('Image from X_train with label '+ str(y_train[935]))
plt.axis('off')
fig.add_subplot(2, 4, 3)
plt.imshow(X_val[0], cmap='gray')
#plt.imshow(X_val[0].transpose((1,0,2)), cmap='gray')
plt.title('Image from X_val with label '+ str(y_val[0]))
plt.axis('off')
fig.add_subplot(2, 4, 4)
plt.imshow(X_val[103], cmap='gray')
#plt.imshow(X_val[103].transpose((1,0,2)), cmap='gray')
plt.title('Image from X_val with label '+ str(y_val[103]))
plt.axis('off')
fig.add_subplot(2, 4, 5)
plt.imshow(X_train_[0], cmap='gray')
plt.title('Image from X_train with label '+ str(y_train_[0]))
plt.axis('off')
fig.add_subplot(2, 4, 6)
plt.imshow(X_train_[935], cmap='gray')
plt.title('Image from X_train with label '+ str(y_train_[935]))
plt.axis('off')
fig.add_subplot(2, 4, 7)
plt.imshow(X_val_[0], cmap='gray')
plt.title('Image from X_val with label '+ str(y_val_[0]))
plt.axis('off')
fig.add_subplot(2, 4, 8)
plt.imshow(X_val_[103], cmap='gray')
plt.title('Image from X_val with label '+ str(y_val_[103]))
plt.axis('off')
plt.show()
```
<p align="justify">Con esta celda lo que se pretende es comprobar que las funciones definidas previamente funcionan como se esperaba.</p>
<img width="1600" height="812" alt="image" src="https://github.com/user-attachments/assets/506b0065-52ae-4c85-85af-de8b99e04201" />

## Construcción del modelo CNN

<p align="justify">El primero de los modelos propuestos emplea una red neuronal convolucional con un solo bloque convolucional y dos capas densas (o Fully Connected), entre las que media una capa "reshape" que se encarga de tomar los datos de la capa convolucional y los preparar para introducirlos en las capas densas.</p>

<p align="justify">Como valores de entrada que le daremos al modelo tendremos las imágenes de los captchas y como salida lo que se obtendrá será la predicción de los caracteres de la etiqueta.</p>

<p align="justify">Como valores a monitorizar tendremos no solo la función loss, si no que también el accuracy, que nos dará la medida de qué tan bien es capaz de clasificar nuestro modelo.</p>

```python
def build_model(): # En lugar de definir una función (que solo usaremos una única vez), se puede
                   # borrar esto y dejar que corra todo por sí solo (quitar indentación).

    # Inputs to the model
    input_img = layers.Input(shape=(200,50,1), name="image", dtype="float32")

    # First conv block
    x = layers.Conv2D(32,(3, 3),activation="relu",kernel_initializer="he_normal",padding="same",name="Conv1")(input_img)
    x = layers.MaxPooling2D((2, 2), name="pool1")(x)

    # Reshape to "split" the volume in 5 time-steps
    x = layers.Reshape(target_shape=(5, 16000), name="reshape")(x)

    # FC layers
    x = layers.Dense(256, activation="relu", name="dense1")(x)

    # Output layer
    output = layers.Dense(19, activation="softmax", name="dense2")(x)

    # Define the model
    model = keras.models.Model(inputs=input_img, outputs=output, name="ocr_classifier_based_model")

    # Compile the model and return
    model.compile(optimizer=keras.optimizers.Adam(), loss="sparse_categorical_crossentropy", metrics=['accuracy'])
    return model # Al quitarle el "def", esto también se ha de eliminar.

# Get the model
model = build_model() # Si le quito el "def", esto se debe borrar.
model.summary()
```
<p align="justify">En la siguiente celda entrenaremos al modelo a lo largo de un número de dado de epochs lo suficientemente grande como para que la precisión del modelo sea la mejor posible.</p>

```python
X_train, X_val, y_train, y_val = create_train_and_validation_datasets(crop=True)
history = model.fit(x=X_train, y=y_train, validation_data=(X_val, y_val), epochs=30)

fig=plt.figure(figsize=(20, 5))
# summarize history for accuracy
fig.add_subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')

# summarize history for loss
fig.add_subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
```
<img width="1621" height="470" alt="image" src="https://github.com/user-attachments/assets/419bb76f-bf4b-4806-a84b-ee07a9505827" />

<p align="justify">Si atendemos a las curvas dibujadas, podremos ver que existe un overfitting muy importante. Esto se puede apreciar al comparar la precisión del modelo cuando estamos entrenándolo frente a cuando lo probamos con datos que no ha visto nunca.</p>

<p align="justify">Pese a tener un desempeño bastante bueno (~80%), es muy mejorable. Sobre todo por la existencia del overfitting, el cual se debe de minimizar lo máximo posible.</p>

<p align="justify">Al crear una segunda capa convolucional, el modelo responde mucho mejor, pues conseguimos reducir más aún las dimensiones del tensor, de tal manera que podemos subdividir mejor los píxeles de la imagen para poder estudiarlas.</p>

<p align="justify">Esto salta de inmediato a la vista al comparar los parámetros totales que teníamos en el modelo anterior frente a lo que obtenemos en este 2º modelo (se reducen a la mitad en este 2º caso).</p>
<img width="1621" height="470" alt="image" src="https://github.com/user-attachments/assets/00b4fd23-12d9-4f92-86fd-4c07d51d067b" />

Ahora vemos que las funciones tienden a los valores óptimos con gran rapidez, necesitando no más de 15 a 20 epochs para llegar a un valor estable.

<p align="justify">También salta a la vista que el modelo posee una precisión mucho mayor tanto en el set de entrenamiento como en el set de validación, llegando a valores por encima del 90% (validación) y del 100% (test). Esto implica una mejora del 8% frente al modelo con una única capa convolucional, donde habíamos obtenido una precisión del 85% frente al 92% obtenido al incluir esta segunda capa.</p>

<p align="justify">También se ha corregido parte del overfitting que acuciábamos antes. Esta corrección, sin embargo, no es todo lo buena que podríamos esperar, por lo que será necesario emplear técnicas de regularización (dropouts) en busca de minimizar esa diferencia entre las 2 curvas.</p>

## Eliminando el Overfitting: Dropouts.

<p align="justify">El concepto de Dropout es, en cierta manera, similar a un Random Forest. Mientras que en un RF tenemos una cantidad elevada de árboles de decisión resolviendo problemas de clasificación hasta que al final poseemos una respuesta como un todo, con los dropouts lo que hacemos es que en cada epoch se seleccionan al azar un número de neuronas o nodos que darán un 0 como salida, de manera que las "desconectaremos" de la red, creando "mini redes".</p>

<p align="justify">Cada una de estas mini redes resolverá el mismo problema, haciendo un assembly al final para poder darnos una serie de resultados que serán muy precisos.</p>

<p align="justify">La idea fundamental es que cada "mini red" no se sobresature con la información que recibe y no le de tiempo a sobreentrenarse, con lo que las predicciones que pudiese hacer a futuro no estuviesen viciadas por esto.</p>

```python
def build_model2(): # En lugar de definir una función (que solo usaremos una única vez), se puede
                    # borrar esto y dejar que corra todo por sí solo (quitar indentación).

    # Inputs to the model
    input_img = layers.Input(shape=(200,50,1), name="image", dtype="float32")

    # First conv block
    x = layers.Conv2D(32,(3, 3),activation="relu",kernel_initializer="he_normal",padding="same",name="Conv1")(input_img)
    x = layers.MaxPooling2D((2, 2), name="pool1")(x)

    # Second conv block
    x = layers.Conv2D(64,(3, 3),activation="relu",kernel_initializer="he_normal",padding="same",name="Conv2")(x)
    x = layers.MaxPooling2D((2, 2), name="pool2")(x)

    # We have used two max pool with pool size and strides 2.
    # Hence, downsampled feature maps are 4x smaller. The number of
    # filters in the last layer is 64 --> output volume shape = (50,12,64)
    # Reshape to "split" the volume in 5 time-steps
    x = layers.Reshape(target_shape=(5, 7680), name="reshape")(x)

    # FC layers
    x = layers.Dense(256, activation="relu", name="dense1")(x) # Deberíamos añadir capas de Dropout
    x= layers.Dropout(0.2)(x)
    x = layers.Dense(64, activation="relu", name="dense2")(x)
    x= layers.Dropout(0.2)(x)

    # Output layer
    output = layers.Dense(19, activation="softmax", name="dense3")(x)

    # Define the model
    model = keras.models.Model(inputs=input_img, outputs=output, name="ocr_classifier_based_model")

    # Compile the model and return
    model.compile(optimizer=keras.optimizers.Adam(), loss="sparse_categorical_crossentropy", metrics=['accuracy'])
    return model # Al quitarle el "def", esto también se ha de eliminar.


# Get the model
model = build_model2() # Si le quito el "def", esto se debe borrar.
model.summary()
```
<img width="1621" height="470" alt="image" src="https://github.com/user-attachments/assets/026e1d28-b765-4ab1-93c8-84413b139c75" />
<p align="justify">En efecto, tal como esperábamos, el dropout ha llevado a mejorar ya no solo la precisión del modelo como tal, si no que ahora se ha llegado a reducir el overfitting hasta prácticamente eliminarlo (sigue habiendo algunas diferencias, pero podríamos decir que entran dentro del margen de error), lo que pone de manifiesto la potencia del dropout.</p>

<p align="justify">Llama la atención también la celeridad con la que se llega a eliminar el subfitting, necesitando solo de 7-8 epochs.</p>

<img width="1570" height="214" alt="image" src="https://github.com/user-attachments/assets/8320b9bf-081e-48c8-bd3e-245dda83f6d0" />

<p align="justify">Tras mostrar los captchas y las predicciones hechas por el modelo, procedemos a calcular la tasa de acierto de este y podemos ver que la precisión a vuelto a mejorar con respecto al modelo anterior, pasando de un 92 a un 97% de precisión.</p>

<p align="justify">Esta mejora, si nos atenemos a los números puros y duros, quizás no parezca muy grande, pero este modelo ya no es tanto la precisión (que también, toda mejora es buena), si no que también ha eliminado el sobreentrenamiento que tenía el modelo anterior.</p>

<p align="justify">Es decir, este modelo no solo es preciso, si no que el día de mañana, cuando se le presenten datos nuevos nunca vistos, será capaz de predecirlos sin miedo a que su entrenamiento pueda viciar los resultados.</p>

## Conclusiones

<p align="justify">Para este proyecto se han creado dos modelos de inteligencia artificial empleando técnicas de Machine Learning y de Deep Learning para poner de manifiesto dos fenómenos:</p>

<p align="justify">1. La capacidad de reconocer una contraseña (un string) y clasificar su fortaleza frente a posibles intentos de hackeo mediante ataques de fuerza bruta.</p>
<p align="justify">2. La capacidad de reconocer imágenes y extraer información de estas para resolver problemas que solo un ser humano debería ser capaz de reconocer.</p>

<p align="justify">En el primer caso, hemos entrenado un modelo clasificador con un dataset de ~700000 entradas, dividiendo el dataset en una proporción 80-20, obteniendo una precisión de predicción y clasificación del 100%, tal y como lo hemos podido comprobar con la matriz de confusión.</p>

<p align="justify">Para el 2º, hemos creado un modelo de red neuronal con una capa convolucional y 2 capas densas y hemos ido mejorándolo poco a poco hasta alcanzar un resultado de predicción superior al 95%.</p>

<p align="justify">Ambos modelos ponen de manifiesto la capacidad que posee la inteligencia artificial y cómo un uso indebido de esta puede poner en riesgo la seguridad de los datos personales de las personas.</p>

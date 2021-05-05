#se importan las modulos necesarios
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import Model
import numpy as np
import matplotlib.pyplot as plt

# Se importan los datos de EMNIST
'''
with open('Datos/Final/train_data.npy', 'rb') as f:
    train_data = np.load(f)

with open('Datos/Final/train_label.npy', 'rb') as f:
    train_label = np.load(f)

with open('Datos/Final/test_data.npy', 'rb') as f:
    test_data = np.load(f)

with open('Datos/Final/test_label.npy', 'rb') as f:
    test_label = np.load(f)
'''

# Se carga el modelo de la red neuronal
model = tf.keras.models.load_model('Modelo/Datos/8N_1O_16NN/model.h5')

# Se genera una nueva "red" en donde la salia son los kernels
features_map = Model(inputs=model.inputs , outputs=model.layers[0].output)

image = []

# Se importan los imágenes generadas a mano
for i in range(97, 123, 1):
    direccion = "Datos/Imagenes/"+chr(i)+".jpg"
    print(direccion)
    img = load_img(direccion , target_size=(28,28), color_mode="grayscale")

    # convert the image to an array
    img = img_to_array(img)
    img /= 255.0
    image.append(img)

# Se selecciona una imagen de cada letra
# Solo se utiliza cuando se usan los datos EMNIST
'''
for i in range(26):
    j = 0
    while j < len(train_data):
        if train_label[j] == i:
            image.append(train_data[j])
            j = len(train_data)-1
        j += 1
'''

# Convierte las imágenes seleccionadas en una matriz numpy
image = np.array(image)

features_map.summary() # Se presenta un resumen de la red neuronal

# Calculando los features_map
features = features_map.predict(image)

# Se imprimen los resultados del feature map
fig = plt.figure(figsize=(10,7))
cont = 1
for j in range(26):
    for i in range(1,features.shape[3]+2):
        plt.subplot(10,9,cont)
        cont += 1
        plt.axis('off')
        if i == 1:
            if j % 10 == 0: plt.title("Original")
            plt.imshow(image[j]*255 , cmap='gray')
        else:
            if j % 10 == 0:
                title = "Mapa " + str(i-1)
                plt.title(title)
            plt.imshow(features[j,:,:,i-2]*255 , cmap='gray')
    if j == 9 or j == 19:
        cont = 1
        plt.show()
        fig = plt.figure(figsize=(10,7))

cont = 1
plt.show()

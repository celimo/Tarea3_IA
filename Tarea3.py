#se importan las modulos necesarios
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

# Se importan los datos seleccionados de EMNIST
with open('Datos/Final/train_data.npy', 'rb') as f:
    train_data = np.load(f)

with open('Datos/Final/train_label.npy', 'rb') as f:
    train_label = np.load(f)

with open('Datos/Final/test_data.npy', 'rb') as f:
    test_data = np.load(f)

with open('Datos/Final/test_label.npy', 'rb') as f:
    test_label = np.load(f)

#se crea el modelo de la red neuronal
model = tf.keras.models.Sequential()

# se añade la primera capa de entrada con imput de 28x28 pixeles de las imágenes
# como son imagenes en escala de grises tiene un valor que va de 0 a 255
# numero de filtros de 12 (son como la cantidad de neuronas de la capa)
#el kernel de (3,3) que es el campo receptivo
#stride 1 para visitar todos los pixeles y padding valid no agrega los zeros en el borde
#funcion de activacion relu
#la funcion del poling es maxpoling

model.add(tf.keras.layers.Conv2D( 8,                          # filtros
                                 (3,3),                        # kernel
                                 padding="same",               # Mismo tamaño de salida
                                 activation = 'relu',          # f activacion
                                 input_shape = (28, 28, 1)))   # input

model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

model.summary()

# se añade una capa conectada de forma densa
# se utiliza la msima cantidad de neuronas que los filtros

model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(16, activation='sigmoid'))

# se usa la capa softmax para clasificar las imágenes
#son 26 letras para clasificar las iamegnes la (ñ) no se toma en cuenta

model.add(tf.keras.layers.Dense(26, activation='softmax'))

model.summary()

optimizador = tf.keras.optimizers.Adam(learning_rate=0.001)

# se compila el modelo
model.compile(optimizer=optimizador,
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# ================ Se entrena el modelo con los datos ================

iteration = 150  # Cantidad de iteraciones para el entrenamiento
CantVal = int(iteration * 0.1)  # Cantidad de validaciones
freqVal = int(iteration / CantVal)  # Frecuencia de las validaciones

# Se inicia  el entrenamiento de la red
# Los primeros parámetros son los datos de entrenamiento y las clasificaciones
# epochs: iteraciones del entrenamiento
# validation_split: Utiliza un 30% de los datos para validación
# validation_freq: cada cierta freqVal de iteraciones se hace la validación
# verbose: (0) No se muestran las iteraciones (1) muestra las iteraciones

# Evitar sobre entrenamiento, se espera "n" iteraciones para comprobar que no crezca
es = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                      mode='min',
                                      verbose=1,
                                      patience=5,
                                      restore_best_weights=True)

training = model.fit(train_data, train_label,
                    epochs=iteration,
                    validation_data=(test_data, test_label),
                    verbose=0,
                    callbacks=[es])


# ================ Configuración para realizar la gráfica ================

loss = training.history['loss']  # Datos de la perida de entrenamiento
val_loss = training.history['val_loss']  # Datos de la perdida de validación
x = []  # Datos eje x para la curva de pérdida

for i in range(len(loss)):  # Se agregan datos al eje x de entrenamiento
    x.append(i)

plt.xlabel("Iteración")
plt.ylabel("Error")
plt.plot(x, loss, label="Pérdida de entrenamiento")
plt.plot(x, val_loss, label="Pérdida de validación")
plt.legend()
plt.show()
#plt.savefig("Graph/CurvasEntrenamiento.png")

# ================ Guardar curvas de périda ================

file = open("curvas/LossTrain.txt", "w")
for i in range(len(loss)):
    file.write(str(x[i]) + " " + str(loss[i]) + "\n")
file.close()

file = open("curvas/ValidTrain.txt", "w")
for i in range(len(val_loss)):
    file.write(str(x[i]) + " " + str(val_loss[i]) + "\n")
file.close()

# ================ Guardar modelo keras ================
model.save('Modelo/model.h5')

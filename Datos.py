from scipy.io import loadmat
import numpy as np

# Se acomodan los datos
data = loadmat(r"letters.mat")

# Preparación del conjunto de datos de entrenamiento
A = data['dataset'][0][0][0][0][0][0] # Contiene los pixeles de las imágenes
B = data['dataset'][0][0][0][0][0][1] # Contiene las etiquetas

# Se hacen una matriz numpy para manipularlos
A = np.array(A)
B = np.array(B)

# Se concatena la información para no generar conflictos al mezclar los dato
C = np.concatenate((A, B), axis = 1)

np.random.seed() # Crea una semilla para el random
# Se mezclan los datos para que sean aleatorios
temp = np.random.permutation(C)

# Arreglo que va a contener las letras
letras = []
cant = 100

# Se recorre 26 veces (26 letras) y se seleccionan 100 letras
for i in range(1, 27, 1):
  cont = 0
  for j in range(len(temp)):
    if temp[j][784] == i:
      letras.append(temp[j])
      cont += 1
    if cont == cant:
      break

# Se genera una matriz numpy para manipularlo
letras = np.array(letras)
# Se mezclan las letras seleccionadas
letras = np.random.permutation(letras)

# La información por su naturaleza se debe permutar
letras2 = letras.T
# Se obtienen ls etiquetas de cada letra en una matriz diferente
etiquetas = np.array([letras2[-1]])
# Por su naturaleza se debe permutar
train_label = etiquetas.T
# Se eliminan las etiquetas para poder darle el formato adecuado
letras2 = np.delete(letras, -1,axis=1)

# Se genera el formato de entrada a la red convolucional
dat_fin = []
for l in range(len(letras2)):
  A2 = letras2[l]
  X = []
  for i in range(0, 784, 28):
    temp = []
    for j in range(28):
      temp.append(A2[i+j])
    X.append(temp)

  X = np.array(X)
  X = X.T

  datos = []
  for i in range(len(X)):
    temp = []
    for j in range(len(X[i])):
      temp2 = []
      for k in range(1):
        temp2.append(X[i][j])
      temp.append(temp2)
    datos.append(temp)

  datos = np.array(datos)

  dat_fin.append(datos)

# Datos finales de entrenamiento
train_data = np.array(dat_fin)

# Normalizar datos
train_data = train_data / 255.0

train_label = train_label - 1

# Preparación del conjunto de datos de validación

A = data['dataset'][0][0][1][0][0][0] # Contiene los pixeles de las imágenes
B = data['dataset'][0][0][1][0][0][1] # Contiene las etiquetas

# Se hacen una matriz numpy para manipularlos
A = np.array(A)
B = np.array(B)

# Se concatena la información para no generar conflictos al mezclar los dato
C = np.concatenate((A, B), axis = 1)

np.random.seed() # Crea una semilla para el random
temp = np.random.permutation(C)

letras = []
# Cantidad de datos para validación
cant = 60

for i in range(1, 27, 1):
  cont = 0
  for j in range(len(temp)):
    if temp[j][784] == i:
      letras.append(temp[j])
      cont += 1
    if cont == cant:
      break

letras = np.array(letras)
letras = np.random.permutation(letras)

letras2 = letras.T
etiquetas = np.array([letras2[-1]])
test_label = etiquetas.T
letras2 = np.delete(letras, -1,axis=1)

dat_fin = []
for l in range(len(letras2)):
  A2 = letras2[l]
  X = []
  for i in range(0, 784, 28):
    temp = []
    for j in range(28):
      temp.append(A2[i+j])
    X.append(temp)

  X = np.array(X)
  X = X.T

  datos = []
  for i in range(len(X)):
    temp = []
    for j in range(len(X[i])):
      temp2 = []
      for k in range(1):
        temp2.append(X[i][j])
      temp.append(temp2)
    datos.append(temp)

  datos = np.array(datos)

  dat_fin.append(datos)

test_data = np.array(dat_fin)

test_data = test_data / 255.0
test_label = test_label - 1

# Cada matriz se debe guardar como un archivo tipo numpy para ser utilizado
# en los entrenamientos y validaciones
with open('Datos/train_data.npy', 'wb') as f:
    np.save(f, train_data)

with open('Datos/train_label.npy', 'wb') as f:
    np.save(f, train_label)

with open('Datos/test_data.npy', 'wb') as f:
    np.save(f, test_data)

with open('Datos/test_label.npy', 'wb') as f:
    np.save(f, test_label)

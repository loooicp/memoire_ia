import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Créez un modèle séquentiel
model = keras.Sequential()

# Ajoutez une couche de convolution 2D avec 32 filtres, une taille de noyau de 3x3 et une fonction d'activation ReLU
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)))

# Ajoutez une couche de pooling max
model.add(layers.MaxPooling2D((2, 2)))

# Ajoutez une autre couche de convolution 2D avec 64 filtres, une taille de noyau de 3x3 et une fonction d'activation ReLU
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

# Ajoutez une autre couche de pooling max
model.add(layers.MaxPooling2D((2, 2)))

# Aplatissez les données en un vecteur
model.add(layers.Flatten())

# Ajoutez une couche dense avec 64 unités et une fonction d'activation ReLU
model.add(layers.Dense(64, activation='relu'))

# Ajoutez la couche de sortie avec autant d'unités que de classes de sortie (par exemple, 10 pour la classification à 10 classes)
model.add(layers.Dense(10, activation='softmax'))

# Compilez le modèle en spécifiant la fonction de perte, l'optimiseur et les métriques
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Affichez un résumé du modèle
model.summary()

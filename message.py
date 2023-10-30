import tensorflow as tf
from tensorflow.keras import layers, models
import mysql.connector
import cv2
import numpy as np

# Étape 1 : Préparation des données

# Établissez une connexion à votre base de données MariaDB locale
conn = mysql.connector.connect(
    host="localhost",
    user="root",
    password="root",
    database="thermic_image_analysis"
)

cursor = conn.cursor()

# Exécutez une requête pour obtenir les chemins des images et les étiquettes (0 pour non rouge, 1 pour rouge)
cursor.execute("SELECT * FROM images;")
data = cursor.fetchall()
conn.close()

# Divisez les données en ensembles d'entraînement et de test (80 % pour l'entraînement, 20 % pour les tests)
data = np.array(data)
np.random.shuffle(data)
split = int(0.8 * len(data))
train_data = data[:split]
test_data = data[split:]

# Fonction pour charger et prétraiter les images
def load_and_preprocess_image(chemin, image_width, image_height):
    image = cv2.imread(chemin)# Normalisation des valeurs de pixel
    image = cv2.resize(image, (image_width, image_height))
    image = image / 255.0  
    return image

train_images = np.array([load_and_preprocess_image(chemin) for chemin, image_width, image_height, _ in train_data])
train_labels = np.array([label for _, label in train_data])
test_images = np.array([load_and_preprocess_image(chemin) for chemin, image_width, image_height, _ in test_data])
test_labels = np.array([label for _, label in test_data])

# Étape 2 : Création de l'architecture du réseau de neurones

image_width = data.image_width
image_height = data.image_height

model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(image_width, image_height, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

# Étape 3 : Compilation du modèle

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Étape 4 : Entraînement du modèle

model.fit(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels))

# Étape 5 : Utilisation du modèle

# Chargez une nouvelle image à partir d'un fichier
new_image = load_and_preprocess_image('nouvelle_image.jpg')

# Utilisez le modèle pour prédire si cette image contient une fuite thermique
prediction = model.predict(np.array([new_image]))

# La valeur de prédiction est un nombre entre 0 et 1, vous pouvez définir un seuil pour déterminer la présence de fuite thermique

if prediction[0] > 0.5:
    print('Fuite thermique détectée !')
else:
    print('Pas de fuite thermique détectée.')
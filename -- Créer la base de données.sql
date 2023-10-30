-- Créer la base de données
CREATE DATABASE thermic_image_analysis;

-- Utiliser la base de données
USE thermic_image_analysis;

-- Créer la table pour les images thermiques 
CREATE TABLE images (
    id INT AUTO_INCREMENT PRIMARY KEY,
    nom_image VARCHAR(255),
    date_enregistrement DATETIME,
    img_link VARCHAR(255),
    img_modify_link(255)
    -- Ajouter d'autres colonnes pour les informations sur l'image
);

-- Créer la table pour les zones rouges
CREATE TABLE zones_rouges (
    id INT AUTO_INCREMENT PRIMARY KEY,
    image_id INT, -- Clé étrangère pour lier à une image 
    x INT,
    y INT,
    largeur INT,
    hauteur INT,
    date_detection DATETIME
);



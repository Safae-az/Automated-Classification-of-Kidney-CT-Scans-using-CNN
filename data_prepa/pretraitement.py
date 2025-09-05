import os
import shutil
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import preprocess_input


data_dir = "data/CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone/CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone"
output_dir = "data/split_data"  # Dossier pour les données séparées
img_size = (224, 224)
batch_size = 32
class_names = ['Normal', 'Cyst', 'Tumor', 'Stone']

# Création des sous-dossiers pour chaque classe
for subset in ['train', 'validation', 'test']:
    for class_name in class_names:
        os.makedirs(os.path.join(output_dir, subset, class_name), exist_ok=True)

# Séparer les images en ensembles train, validation et test
for class_name in class_names:
    class_folder = os.path.join(data_dir, class_name)
    images = os.listdir(class_folder)
    
    # Séparer les données : 70% train, 15% validation, 15% test
    train_val_images, test_images = train_test_split(images, test_size=0.15, random_state=42)
    train_images, val_images = train_test_split(train_val_images, test_size=0.15, random_state=42)

    # Déplacer les images dans les bons sous-dossiers
    for image in train_images:
        shutil.move(os.path.join(class_folder, image), os.path.join(output_dir, 'train', class_name, image))
    for image in val_images:
        shutil.move(os.path.join(class_folder, image), os.path.join(output_dir, 'validation', class_name, image))
    for image in test_images:
        shutil.move(os.path.join(class_folder, image), os.path.join(output_dir, 'test', class_name, image))

print("Séparation des données terminée.")

# Prétraitement des images
print("\nPrétraitement des images")

# Générateur pour l'augmentation des données d'entraînement
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=15,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],
    zoom_range=0.1,
    validation_split=0.3  # 70% entraînement, 30% test/val
)

# Générateur pour validation/test (juste normalisation)
test_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    validation_split=0.5  # 15% validation, 15% test
)

# Chargement des données d'entraînement (70% du total)
train_generator = train_datagen.flow_from_directory(
    os.path.join(output_dir, 'train'),
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True
)

# Chargement des données de validation (15% du total)
validation_generator = test_datagen.flow_from_directory(
    os.path.join(output_dir, 'validation'),
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

# Chargement des données de test (15% du total)
test_generator = test_datagen.flow_from_directory(
    os.path.join(output_dir, 'test'),
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

# Affichage des exemples augmentees
print("\nAffichage d'exemples d'augmentation de données...")

vis_datagen = ImageDataGenerator(
    rotation_range=15,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],
    zoom_range=0.1
)

vis_generator = vis_datagen.flow_from_directory(
    os.path.join(output_dir, 'train'),
    target_size=img_size,
    batch_size=1,
    class_mode='categorical',
    shuffle=True
)

plt.figure(figsize=(12, 8))
for i in range(9):
    plt.subplot(3, 3, i + 1)
    image, label = next(vis_generator)
    plt.imshow(image[0].astype(np.uint8))
    label_idx = np.argmax(label[0])
    plt.title(f"Classe: {class_names[label_idx]}")
    plt.axis('off')

plt.tight_layout()
plt.savefig('exemples_augmentation.png')
plt.close()

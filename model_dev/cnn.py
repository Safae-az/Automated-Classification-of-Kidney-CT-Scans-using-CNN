import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt

def build_and_train_model(data_dir, img_size=(224, 224), num_classes=4, batch_size=32, epochs=5):
    # Création du modele
    print("\n Création du modèle de base (CNN simple)")
    model = Sequential()

    # Bloc 1
    model.add(Conv2D(32, (3, 3), padding='same', input_shape=(img_size[0], img_size[1], 3)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Bloc 2
    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Bloc 3
    model.add(Conv2D(128, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Bloc 4
    model.add(Conv2D(256, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Couches fully connected
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))  
     # Couche de sortie avec softmax
    model.add(Dense(num_classes, activation='softmax')) 

    # Compilation du modèle
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    print(model.summary())

    # Préparation des données
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=15,
        horizontal_flip=True,
        brightness_range=[0.8, 1.2],
        zoom_range=0.1,
        validation_split=0.2  # 20% validation, 80% train
    )

    # Générateur pour l'entraînement (80% des données)
    train_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='training',
        shuffle=True
    )

    # Générateur pour la validation (20% des données)
    validation_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation',
        shuffle=False
    )

    # Entraînement du modèle sur un sous-ensemble des données
    print("\nEntraînement du modèle sur un sous-ensemble des données...")
    model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // batch_size,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // batch_size
    )

    # Sauvegarde du modèle
    model.save('model_base.h5')

    return model


data_dir = "data\\split_data\\train"  
model = build_and_train_model(data_dir)


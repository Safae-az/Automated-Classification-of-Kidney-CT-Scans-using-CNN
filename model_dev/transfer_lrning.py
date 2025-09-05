from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import matplotlib.pyplot as plt

# 1. Construction du modèle avec architecture personnalisable
def build_model(input_shape=(224, 224, 3), num_classes=4, dropout_rate=0.5):
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(dropout_rate)(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    return Model(inputs=base_model.input, outputs=predictions), base_model

# 2. Générateurs avec augmentations modifiables
def get_data_generators(data_path, img_size=(224, 224), batch_size=32, validation_split=0.3):
    datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        shear_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=validation_split
    )
    
    train_gen = datagen.flow_from_directory(
        data_path + '/train',
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='training',
        shuffle=True
    )
    
    val_gen = datagen.flow_from_directory(
        data_path + '/val',
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation',
        shuffle=False
    )
    return train_gen, val_gen

# 3. Entraînement complet avec hyperparamètres ajustables
def train(data_path, model_save_path='kidney_model.h5', img_size=(224, 224), batch_size=32,
          lr_phase1=1e-3, lr_phase2=1e-4, fine_tune_layers=20, dropout_rate=0.5,
          epochs_phase1=15, epochs_phase2=10):
    
    train_gen, val_gen = get_data_generators(data_path, img_size, batch_size)
    model, base_model = build_model(img_size + (3,), dropout_rate=dropout_rate)

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6),
        ModelCheckpoint(model_save_path, monitor='val_accuracy', save_best_only=True)
    ]
    
    print(" Phase 1 : Entraînement des nouvelles couches")
    base_model.trainable = False
    model.compile(optimizer=Adam(learning_rate=lr_phase1), loss='categorical_crossentropy', metrics=['accuracy'])
    history1 = model.fit(train_gen, validation_data=val_gen, epochs=epochs_phase1, callbacks=callbacks)
    
    print("\n Phase 2 : Fine-tuning des derniers blocs")
    base_model.trainable = True
    for layer in base_model.layers[:-fine_tune_layers]:
        layer.trainable = False

    model.compile(optimizer=Adam(learning_rate=lr_phase2), loss='categorical_crossentropy', metrics=['accuracy'])
    history2 = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs_phase1 + epochs_phase2,
        initial_epoch=history1.epoch[-1] + 1,
        callbacks=callbacks
    )

    model.save(model_save_path)
    print(" Modèle sauvegardé dans :", model_save_path)
    return model, history1, history2


if __name__ == "__main__":
    train('C:/Users/lenovo/Desktop/machine learning project/split_dataset',
          model_save_path='kidney_model.h5',
          batch_size=32,
          lr_phase1=0.001,
          lr_phase2=0.0001,
          dropout_rate=0.5,
          fine_tune_layers=20)

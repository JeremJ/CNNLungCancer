import tensorflow as tf
from keras.callbacks import EarlyStopping
from keras.layers import Dense, Flatten, MaxPooling2D, Conv2D
from keras.models import Sequential
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

from model_commons import load_data, plot_model_results

#należy wskazać odpowiednią ścieżkę do folderu po pobraniu projektu
TRAIN_DIR = 'F:/praca-magisterska/CNN-github/CNNLungCancer/Dane/trenujace/semi-segmentation/'
TEST_DIR = 'F:/praca-magisterska/CNN-github/CNNLungCancer/Dane/testowe/semi-segmentation/'
VAL_DIR = 'F:/praca-magisterska/CNN-github/CNNLungCancer/Dane/walidujace/semi-segmentation/'
IMG_SIZE = [224, 224]

X_train, y_train, labels = load_data(TRAIN_DIR, IMG_SIZE)
X_test, y_test, _ = load_data(TEST_DIR, IMG_SIZE)
X_val, y_val, _ = load_data(VAL_DIR, IMG_SIZE)

model = Sequential([
    Conv2D(16, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D(2, 2),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(1, activation='sigmoid')])

model.summary()
model.compile(
    loss='binary_crossentropy',
    metrics=['accuracy'],
    optimizer=Adam(learning_rate=1e-3)
)
RANDOM_SEED = 123

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=40,
    brightness_range=[0.5, 1.5],
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(
    rescale=1. / 255
)

test_datagen = ImageDataGenerator(
    rescale=1. / 255
)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    color_mode='rgb',
    target_size=IMG_SIZE,
    batch_size=32,
    class_mode='binary',
    shuffle=True,
    seed=RANDOM_SEED
)

test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    color_mode='rgb',
    target_size=IMG_SIZE,
    batch_size=8,
    class_mode='binary',
    seed=RANDOM_SEED
)

validation_generator = val_datagen.flow_from_directory(
    VAL_DIR,
    color_mode='rgb',
    target_size=IMG_SIZE,
    batch_size=16,
    class_mode='binary',
    seed=RANDOM_SEED
)

es = EarlyStopping(
    monitor='val_accuracy',
    mode='max',
    patience=6
)

history = model.fit(
    train_generator,
    steps_per_epoch=50,
    epochs=10,
    validation_data=validation_generator,
    validation_steps=30
)

plot_model_results(model.history)
result = model.evaluate(test_generator, batch_size=128)
print("test_loss, test accuracy", result)

predictions = model.predict(X_test).round()

print(classification_report(y_test, predictions))
print(confusion_matrix(y_test, predictions))

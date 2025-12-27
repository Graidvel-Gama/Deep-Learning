import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.applications.efficientnet import preprocess_input
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import cv2

DATA_DIR = "./Dataset/dataset"
IMG_SIZE = (224, 224)
BATCH_SIZE = 16

DS = image_dataset_from_directory(
    DATA_DIR,
    labels = 'inferred',
    label_mode = 'categorical',
    batch_size = BATCH_SIZE,
    image_size = IMG_SIZE,
    shuffle = True,
    seed = 42
)

class_names = DS.class_names

print(class_names)
def to_binary_label(one_hot, class_names):
    idx = np.argmax(one_hot)
    name = class_names[idx]
    return 0 if name.startswith("fresh") else 1
images = []
labels = []

for batch_images, batch_labels in DS:
    batch_labels_binary = np.array([0 if class_names[np.argmax(l.numpy())].startswith("fresh") else 1 for l in batch_labels])
    images.append(batch_images.numpy())
    labels.append(batch_labels_binary)

x = np.concatenate(images, axis = 0)
y = np.concatenate(labels, axis = 0)
x = preprocess_input(x)

x_train, x_temp, y_train, y_temp = train_test_split(x, y, train_size = 0.7, stratify = y, random_state = 42)
x_test, x_val, y_test, y_val = train_test_split(x_temp, y_temp, test_size = 0.5, stratify = y_temp, random_state = 42)

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.2),
    layers.RandomContrast(0.1)
])

EffNet = EfficientNetB0(
    include_top = False,
    weights = 'imagenet',
    input_shape = (IMG_SIZE[0], IMG_SIZE[1], 3)
)

EffNet.trainable = True

for layer in EffNet.layers[:-20]:
    layer.trainable = False

Model = models.Sequential([
    layers.Input(shape = (224, 224, 3)),
    data_augmentation,
    EffNet,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.2),
    layers.Dense(1, activation = 'sigmoid')
])

early_stop = EarlyStopping(monitor = 'val_loss', patience = 5, restore_best_weights = True)
checkpoint = ModelCheckpoint("model/best_model.h5", monitor='val_loss', save_best_only = True)
Model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = 0.0001), loss = 'binary_crossentropy', metrics = ['accuracy'])
history = Model.fit(x_train, y_train, validation_data = (x_val, y_val), epochs = 20, batch_size = BATCH_SIZE, callbacks = [early_stop, checkpoint])

y_pred_prob = Model.predict(x_test)
y_pred = (y_pred_prob < 0.5).astype(int)
acc = accuracy_score(y_test, y_pred)
pres = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
F1 = f1_score(y_test, y_pred)

print(f"Accuracy Score  : {acc}")
print(f"Precision Score : {pres}")
print(f"Recall Score    : {recall}")
print(f"F1 Score        : {F1}")

import pickle
with open("history.pkl", "wb") as f:
    pickle.dump(history.history, f)

y_pred_prob = Model.predict(x_test)
y_pred = (y_pred_prob < 0.5).astype(int)
acc = accuracy_score(y_test, y_pred)
pres = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
F1 = f1_score(y_test, y_pred)

print(f"Accuracy Score  : {acc}")
print(f"Precision Score : {pres}")
print(f"Recall Score    : {recall}")
print(f"F1 Score        : {F1}")

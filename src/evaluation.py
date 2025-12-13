import pickle
import matplotlib.pyplot as plt
import tensorflow as tf


with open('history.pkl', 'rb') as file:
    history = pickle.load(file)

model = tf.keras.models.load_model("model/best_model.h5")

plt.figure(figsize=(20, 8))
plt.subplot(1, 2, 1)
plt.plot(history["accuracy"], label = "Tranining Accuracy")
plt.plot(history["val_accuracy"], label = "Validation Accuracy")
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(history["loss"], label = "Tranining Loss")
plt.plot(history["val_loss"], label = "Validation Loss")
plt.legend()
plt.show()


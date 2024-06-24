import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf

def clear_screen():
    # Controlla il sistema operativo e usa il comando appropriato
    if os.name == 'nt':  # Per Windows
        os.system('cls')
    else:  # Per Unix/Linux/Mac
        os.system('clear')


mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Definisci il modello usando un livello Input all'inizio
model = tf.keras.models.Sequential([
  tf.keras.layers.InputLayer(input_shape=(28, 28)),  # Usa InputLayer
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10)
])

clear_screen()
print("TensorFlow version:", tf.__version__)

predictions = model(x_train[:1]).numpy()
print("===== PREDICTIONS =====")
print(predictions)
print("\n")

print("===== SOFTMAX =====")
print(tf.nn.softmax(predictions).numpy())
print("\n")

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

print("===== LOSS =====")
print(loss_fn(y_train[:1], predictions).numpy())
print("\n")

model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)

model.evaluate(x_test,  y_test, verbose=2)

probability_model = tf.keras.Sequential([
  model,
  tf.keras.layers.Softmax()
])

probability_model(x_test[:5])

print(model)
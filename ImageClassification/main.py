import tensorflow as tf
import os
from plotFile import plot_image, plot_value_array

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

def clear_screen():
    # Controlla il sistema operativo e usa il comando appropriato
    if os.name == 'nt':  # Per Windows
        os.system('cls')
    else:  # Per Unix/Linux/Mac
        os.system('clear')

clear_screen()

# Versione di TensorFlow utilizzata
print("TensorFlow version: ", tf.__version__)

#** ===== CARICAMENTO DEL DATASET =====
# Carica il dataset Fashion MNIST per effettuare l'addestramento ovvero un dataset di 70.000 immagini in scala di grigi di vestiti.
fashion_mnist = tf.keras.datasets.fashion_mnist

# Gli array train_images e train_labels sono i dati che il modello usa per imparare.
# Successivamente il modello verrà testato con l'array test_images e test_labels.
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# Ogni immagine è mappata su una singola etichetta. Dato che i nomi delle classi non sono inclusi nel dataset,
# li memorizziamo qui per usarli successivamente durante la visualizzazione.
# Le etichette sono: 
# 0 = T-shirt/top, 1 = Trouser, 2 = Pullover, 3 = Dress, 4 = Coat, 5 = Sandal, 6 = Shirt, 7 = Sneaker, 8 = Bag, 9 = Ankle boot 
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Array contenenti le immagini per ADDESTRARE il modello
print("\n")
print("===== TRAIN IMAGES =====")
print("N. of Imgs - Width - Heigth\n", train_images.shape)
print("\n")
print("===== TRAIN LABELS =====")
print("N. of Labels\n", len(train_labels))
print("\n")
# Array contenenti le immagini per TESTARE il modello
print("===== TEST IMAGES =====")
print("N. of Imgs - Width - Heigth\n", test_images.shape)
print("\n")
print("===== TEST LABELS =====")
print("N. of Labels\n", len(test_labels))
print("\n")

#** ===== PREEELABORAZIONE DEI DATI =====
# Prima di addestrare la rete neurale, è necessario preelaborare i dati.
# Se si esaminano le prime immagini del set di addestramento, si vedrà che i valori dei pixel sono compresi nell'intervallo da 0 a 255.
'''
plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show()
'''
# Prima di alimentare la rete neurale, è necessario ridurre questi valori a un intervallo da 0 a 1.
# Per fare ciò, si dividono i valori per 255. È importante che il set di addestramento e il set di test vengano preelaborati nello stesso modo.
train_images = train_images / 255.0
test_images = test_images / 255.0

# Per verificare che i dati siano nel formato corretto e che siano pronti per costruire e addestrare la rete neurale,
# visualizziamo le prime 25 immagini dal set di addestramento e visualizziamo il nome della classe sotto ciascuna immagine.
'''
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()
'''

#** ===== COSTRUZIONE DEL MODELLO =====
#* === 1. CONFIGURAZIONE DEI LIVELLI (LAYERS) ===
# Il blocco di costruzione di base di una rete neurale è il livello. I livelli estraggono rappresentazioni dai dati inseriti in essi,
# con la speranza che queste rappresentazioni siano significative per il problema in esame.
# La maggior parte del deep learning consiste nel concatenare insieme semplici livelli. 
# La maggior parte dei livelli, ad esempio tf.keras.layers.Dense , ha parametri che vengono appresi durante l'allenamento.
model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(28, 28)), # Usa InputLayer per specificare la forma dei dati di input (un array bidimensionale 28x28 px)
    tf.keras.layers.Flatten(), # Layer Flatten: trasforma il formato delle immagini da array bidimensionale (28x28 px) ad array monodimensionale (28*28=784px) (in breve schiaccia la matrice)
    tf.keras.layers.Dense(128, activation='relu'), # Layer Dense: Strato di 128 neuroni (o nodi) densamente connessi tra di loro. 
    tf.keras.layers.Dense(10) # Layer Dense: ritorna un array logit con lunghezza 10. Ogni nodo contiene un punteggio che indica che l'immagine corrente appartiene a una delle 10 classi.
])

#* === 2. COMPILAZIONE DEL MODELLO ===
# Prima che il modello sia pronto per l'addestramento, è necessario alcune impostazioni in più.
# Queste sono aggiunte durante la fase di compilazione del modello:
# - Ottimizzatore: Indica come il modello viene aggiornato in base ai dati che vede e alla funzione loss.
# - Funzione Loss: Misura quanto accurato è il modello durante l'addestramento. L'obiettivo è minimizzare questa funzione per "guidare" il modello nella direzione corretta.
# - Metriche: Vengono utilizzate per monitorare le fasi di addestramento e test. L'esempio seguente utilizza l'accuratezza, la frazione delle immagini che sono state classificate correttamente.
model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])


#** ===== ADDESTRAMENTO DEL MODELLO =====
# Addestrare la rete neurale richiede i seguenti passaggi:
# 1. Alimentare il modello con i dati di addestramento. In questo esempio, i dati di addestramento sono gli array train_images e train_labels.
# 2. Il modello impara ad associare le immagini alle etichette.
# 3. Chiedere al modello di fare previsioni su un set di test, in questo esempio l'array test_images.
# 4. Verificare che le previsioni corrispondano alle etichette dal test_labels array.
model.fit(train_images, train_labels, epochs=10)


#** ===== VALUTAZIONE DEL MODELLO =====
# Successivamente, confrontiamo come si comporta il modello sui dati di test:
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
print('\nTest accuracy:', test_acc)


#** ===== PREDIZIONI =====
# Con il modello addestrato, possiamo usarlo per fare previsioni su alcune immagini.
# Aggiungiamo un livello softmax al modello, per convertire i logit (output lineari del modello) in probabilità, che sono più facili da interpretare.
probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])

# Ora, facciamo previsioni. Questo modello predice l'etichetta di ogni immagine nel set di test.
predictions = probability_model.predict(test_images)
# Questo output è un array di 10 numeri. Rappresentano la "confidenza" del modello che l'immagine corrisponde a ciascuna delle 10 diverse classi di abbigliamento.
print("===== PREDICTIONS =====")
print("First Prediction:\n", predictions[0])
print("Highest Confidence Prediction:\n", np.argmax(predictions[0]), " - ", class_names[np.argmax(predictions[0])])
print("\n")

#* === VERIFICA DELLE PREDIZIONI ===
# Possiamo visualizzare il risultato della previsione in modo più intuitivo, ovvero graficamente.
num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
    plt.subplot(num_rows, 2*num_cols, 2*i+1)
    plot_image(i, predictions[i], test_labels, test_images, class_names)
    plt.subplot(num_rows, 2*num_cols, 2*i+2)
    plot_value_array(i, predictions[i], test_labels)
plt.tight_layout()
plt.show()

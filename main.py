import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
import os
import tensorflow as tf

# Configurar uso de la GPU en TensorFlow
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Configuración inicial
num_classes = 3  # Piedra -> 0, Papel -> 1, Tijera -> 2

# Crear datos de ejemplo
X_data = np.random.randint(3, size=(1000, 2))  # Dos entradas (jugada de usuario y máquina)
y_data = (X_data[:, 0] - X_data[:, 1]) % 3  # Calculamos el resultado esperado
y_data = to_categorical(y_data, num_classes=num_classes)

# Dividir en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, random_state=42)

# Configuración para guardar y cargar el modelo
model_path = "piedra_papel_tijera_model.keras"

# Verificar si el modelo guardado existe para cargarlo
if os.path.exists(model_path):
    model = load_model(model_path)
    print("Modelo cargado desde disco.")
else:
    # Definir y compilar el modelo si no existe previamente guardado
    model = Sequential([
        Dense(64, activation='relu', input_shape=(2,)),
        Dense(32, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Callbacks para evitar el sobreajuste
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),
        ModelCheckpoint(model_path, save_best_only=True, monitor='val_loss', verbose=1)
    ]

    # Entrenar el modelo con callbacks
    model.fit(X_train, y_train, epochs=50, batch_size=16, validation_data=(X_test, y_test), callbacks=callbacks)
    print("Modelo entrenado y guardado en disco.")

# Evaluar el modelo
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Precisión del modelo: {accuracy:.2f}")

# Función para jugar piedra, papel o tijera
def jugar_piedra_papel_tijera(jugada_usuario):
    jugadas = ["Piedra", "Papel", "Tijera"]
    jugada_maquina = np.random.randint(3)
    print(f"Tu jugada: {jugadas[jugada_usuario]}, Jugada de la máquina: {jugadas[jugada_maquina]}")
    
    # Predecir el resultado
    prediccion = model.predict(np.array([[jugada_usuario, jugada_maquina]]), verbose=0)
    resultado = np.argmax(prediccion)
    
    if resultado == 0:
        print("Empate.")
    elif resultado == 1:
        print("Ganaste.")
    else:
        print("Perdiste.")

# Ejemplo de uso
jugar_piedra_papel_tijera(2)  # 0 para Piedra, 1 para Papel, 2 para Tijera

# piedra-papel-tijera

Cambios y optimizaciones:

Configuración GPU: Configura TensorFlow para usar la GPU si está disponible. También habilita el crecimiento de memoria para evitar que se reserve más de la necesaria.
Modelo simplificado: Utiliza un solo Sequential con una configuración concisa para mantener la estructura del código clara.
Verbosidad del predict: La predicción se realiza en silencio (verbose=0), útil para reducir el ruido al jugar.
EarlyStopping y ModelCheckpoint: Evitan el sobreentrenamiento y optimizan la precisión guardando solo los mejores modelos.

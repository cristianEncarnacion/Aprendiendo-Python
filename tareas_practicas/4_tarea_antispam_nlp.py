import tensorflow as tf
import numpy as np

# 1. DATOS DE ENTRENAMIENTO
lista_mensajes = [
    "hola mamá llego tarde a cenar",                # 0 (Normal)
    "FELICIDADES ganaste un iphone haz clic aqui",  # 1 (Spam)
    "no olvides comprar leche en el super",         # 0 (Normal)
    "CREDITO APROBADO deposita 500 dolares ahora",  # 1 (Spam)
    "nos vemos mañana en la oficina saludos",       # 0 (Normal)
    "URGENTE tu cuenta ha sido bloqueada ingresa ya"# 1 (Spam)
]

# Convertimos la lista a un Tensor nativo
mensajes = tf.constant(lista_mensajes)
etiquetas = np.array([0, 1, 0, 1, 0, 1], dtype=float)

# 2. EL TRADUCTOR
vectorizador = tf.keras.layers.TextVectorization(max_tokens=50, output_sequence_length=8)
vectorizador.adapt(mensajes)

# 3. CONSTRUIMOS EL MODELO
modelo=tf.keras.Sequential([
    vectorizador, # Capa de traducción de texto a números (IDs)
    tf.keras.layers.Embedding(input_dim=50, output_dim=8), # Capa de mapa de significados
    tf.keras.layers.GlobalAveragePooling1D(), # Capa de promedio para obtener un vector fijo
    tf.keras.layers.Dense(units=16, activation='relu'), # Capa oculta con ReLU
    tf.keras.layers.Dense(units=1, activation='sigmoid') # Capa de salida para clasificación binaria
])

modelo.compile(optimizer=tf.keras.optimizers.Adam(0.01), loss='binary_crossentropy')
print("Entrenando el modelo antispam...")
modelo.fit(mensajes, etiquetas, epochs=100, verbose=0)
print("¡Entrenamiento completado!")
# 4. PREDICCIÓN CON UN MENSAJE NUEVO
nuevo_mensaje = tf.constant(["hola ganaste un premio urgente haz clic"])
resultado = modelo.predict(nuevo_mensaje)
print("Probabilidad de que el mensaje sea SPAM: ", resultado[0][0])
# Congelamos el modelo y lo guardamos en un archivo físico
modelo.save("mi_detector_spam.keras")
print("¡Modelo guardado exitosamente en el disco duro!")
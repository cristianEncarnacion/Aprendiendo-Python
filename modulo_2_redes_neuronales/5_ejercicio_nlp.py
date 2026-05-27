import tensorflow as tf
import numpy as np

# 1. NUESTROS DATOS (El texto crudo)
lista_reseñas = [
    "me encantó la película es maravillosa", # 1 (Positivo)
    "es horrible y muy aburrida",            # 0 (Negativo)
    "la mejor película me encantó",          # 1 (Positivo)
    "pésima película horrible",              # 0 (Negativo)
    "maravillosa y muy buena",               # 1 (Positivo)
    "muy aburrida y pésima"                  # 0 (Negativo)
]

# ¡LA SOLUCIÓN!: Convertimos la lista de texto a un Tensor nativo de TensorFlow
reseñas = tf.constant(lista_reseñas)
etiquetas = np.array([1, 0, 1, 0, 1, 0], dtype=float)

# 2. CREAMOS EL TRADUCTOR
# max_tokens=100 (aprenderá hasta 100 palabras distintas)
# output_sequence_length=6 (cortará o rellenará las frases para que todas midan 6 palabras exactas)
vectorizador = tf.keras.layers.TextVectorization(max_tokens=100, output_sequence_length=6)
vectorizador.adapt(reseñas) # Aquí el traductor "lee" los tensores y arma su diccionario

# 3. CONSTRUIMOS EL MODELO
modelo = tf.keras.Sequential([
    vectorizador, # Primera capa: Traduce el texto a números (IDs)
    
    # Segunda capa: El mapa de significados (16 dimensiones para 100 palabras)
    tf.keras.layers.Embedding(input_dim=100, output_dim=16),
    
    # Tercera capa: Saca el promedio del significado de toda la frase
    tf.keras.layers.GlobalAveragePooling1D(),
    
    # Cuarta capa: El cerebro analítico tradicional
    tf.keras.layers.Dense(units=16, activation='relu'),
    
    # Quinta capa: La decisión final (1 o 0)
    tf.keras.layers.Dense(units=1, activation='sigmoid')
])

# 4. COMPILAMOS (Igual que siempre para problemas de Sí/No)
modelo.compile(optimizer=tf.keras.optimizers.Adam(0.01), loss='binary_crossentropy')

# 5. ENTRENAMIENTO
print("Entrenando IA de lectura...")
modelo.fit(reseñas, etiquetas, epochs=100, verbose=0) 
print("¡Entrenamiento completado!")

# 6. PREDICCIÓN FINAL
# También convertimos la frase nueva a un Tensor usando tf.constant
nueva_reseña = tf.constant(["la película es maravillosa me encantó"])

resultado = modelo.predict(nueva_reseña)
print("Probabilidad de que la reseña sea POSITIVA: ", resultado[0][0])
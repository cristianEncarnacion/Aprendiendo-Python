import tensorflow as tf
import numpy as np # NumPy es una biblioteca de Python para el cálculo numérico y la manipulación de matrices.


# 1. Definimos nuestrso datos de entrenamiento (Tamaño, Baños, Color)

caracteristicas=np.array([[100,2,1], # Casa 1: 100m2, 2 baños, puerta roja (1)
[150, 3, 0],  # Casa 2: 150m2, 3 baños, puerta blanca (0)
    [80,  1, 1],  # Casa 3: 80m2, 1 baño, puerta roja (1)
    [200, 4, 0]], dtype=float) # Casa 4: 200m2, 4 baños, puerta blanca (0)

#Etiquetas (Los precios reales de la casa en miles de dólares)
precios=np.array([120,180,90,250],dtype=float)

# 2. Construimos la Red Neuronal
# tf.Keras.sequential es una forma de construir un modelo de red neuronal en TensorFlow. Permite apilar capas de manera secuencial, lo que es útil para modelos simples.
# tf.keras.layers.Dense es una capa densa (fully connected) que conecta cada neurona de la capa anterior con cada neurona de la capa actual. En este caso, estamos creando una capa de salida con 1 neurona (para predecir el precio) y especificando que la capa de entrada tiene 3 características (tamaño, baños, color).
modelo=tf.keras.Sequential([
    tf.keras.layers.Dense(units=1, input_shape=[3]) # Capa de salida con 1 neurona y 3 entradas
])


#3. Compilamos el modelo
modelo.compile(optimizer=tf.keras.optimizers.Adam(0.1), loss='mean_squared_error') # Compilamos el modelo con el optimizador Adam y la función de pérdida de error cuadrático medio. El optimizador Adam es un algoritmo de optimización que ajusta los pesos de la red neuronal durante el entrenamiento para minimizar la función de pérdida. La función de pérdida de error cuadrático medio (mean_squared_error) mide la diferencia entre los valores predichos por el modelo y los valores reales, penalizando más las predicciones que están más lejos de los valores reales.

#4. Entrenamiento
print("Entrenando el modelo...")
modelo.fit(caracteristicas,precios,epochs=500, verbose=0) # Entrenamos el modelo con los datos de características y precios durante 500 épocas. El parámetro verbose=0 se utiliza para suprimir la salida de información durante el entrenamiento.


# 5. PREDICCIÓN CON DATOS NUEVOS
nueva_casa = np.array([[120, 2, 0]], dtype=float) # Nueva casa: 120m2, 2 baños, puerta blanca (0)
resultado = modelo.predict(nueva_casa)

print("El precio estimado de la nueva casa es: $", resultado[0][0], "mil dólares")
print("Modelo entrenado.")
import tensorflow as tf
import numpy as np



# 1. Definimos caracteristicas y etiquetas

etiqueta=np.array([40,70,120,35], dtype=float)

caracteristicas=np.array([[2,1],
    [5,2],
    [10,3],
    [1,1]],dtype=float)

# 2. Construimos la Red Neuronal
modelo=tf.keras.Sequential([
    tf.keras.layers.Dense(units=1, input_shape=[2])
])


#3. Compilamos el modelo
modelo.compile(optimizer=tf.keras.optimizers.Adam(0.1), loss='mean_squared_error')

#4. Entrenamiento
print("Entrenando el modelo...")
modelo.fit(caracteristicas,etiqueta,epochs=500, verbose=0)


# 5. PREDICCIÓN CON DATOS NUEVOS
nuevo_candidato=np.array([[7,2]],dtype=float)

resultado=modelo.predict(nuevo_candidato)
print("El salario estimado del nuevo candidato es: $", resultado[0][0], "mil dólares")
print("Modelo entrenado.")
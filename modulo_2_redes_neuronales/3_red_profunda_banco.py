import tensorflow as tf
import numpy as np

# 1. DATOS DE ENTRENAMIENTO
# Características: [Edad, Salario en miles]
caracteristicas = np.array([
    [25, 2.0],  # Cliente 1: 25 años, gana 2k -> No paga
    [45, 5.0],  # Cliente 2: 45 años, gana 5k -> Sí paga
    [30, 1.5],  # Cliente 3: 30 años, gana 1.5k -> No paga
    [50, 8.0],  # Cliente 4: 50 años, gana 8k -> Sí paga
    [35, 4.0],  # Cliente 5: 35 años, gana 4k -> Sí paga
    [22, 1.0]   # Cliente 6: 22 años, gana 1k -> No paga
], dtype=float)

# Etiquetas: 0 = No paga (Riesgo), 1 = Sí paga (Aprobado)
etiquetas = np.array([0, 1, 0, 1, 1, 0], dtype=float)

# 2. CONSTRUCCIÓN DE LA RED NEURONAL

modelo=tf.keras.Sequential([
    tf.keras.layers.Input(shape=(2,)), # Especificamos que la entrada tiene 2 características (Edad y Salario)
    tf.keras.layers.Dense(units=16,activation='relu'), # Capa oculta con 16 neuronas y función de activación ReLU: ReLU (Rectified Linear Unit) es una función de activación que introduce no linealidad en el modelo, permitiendo que la red neuronal aprenda relaciones complejas entre las características de entrada y las etiquetas. La función ReLU devuelve el valor de entrada si es positivo y 0 si es negativo, lo que ayuda a evitar el problema del gradiente desvanecido durante el entrenamiento.
    tf.keras.layers.Dense(units=8,activation='relu'), # Capa oculta con 8 neuronas y función de activación ReLU: ReLU (Rectified Linear Unit) es una función de activación que introduce no linealidad en el modelo, permitiendo que la red neuronal aprenda relaciones complejas entre las características de entrada y las etiquetas. La función ReLU devuelve el valor de entrada si es positivo y 0 si es negativo, lo que ayuda a evitar el problema del gradiente desvanecido durante el entrenamiento.
    tf.keras.layers.Dense(units=1,activation='sigmoid') # Capa de salida con 1 neurona y función de activación sigmoide para clasificación binaria
])


modelo.compile(optimizer=tf.keras.optimizers.Adam(0.01), loss="binary_crossentropy") # Compilamos el modelo con el optimizador Adam y la función de pérdida de error cuadrático medio.
print("Entrenando el modelo...")
modelo.fit(caracteristicas,etiquetas,epochs=500,verbose=0)

# 3. PREDICCIÓN CON DATOS NUEVOS
nuevo_cliente=np.array([[32,3.5]],dtype=float) # Nuevo cliente: 32 años, gana 3.5k
resultado=modelo.predict(nuevo_cliente)
print("Probabilidad de que el nuevo cliente pague el préstamo: ", resultado[0][0])
if resultado[0][0] > 0.5:
    print("El modelo predice que el nuevo cliente pagará el préstamo.")
else:    print("El modelo predice que el nuevo cliente no pagará el préstamo.")
print("Modelo entrenado.")
import tensorflow as tf
import numpy as np

# ==========================================
# 1. GENERACIÓN DE "BIG DATA"
# ==========================================
print("Generando 1000 días de historial de ventas...")
np.random.seed(42) # Para que a ambos nos den los mismos números aleatorios
tiempo = np.arange(1000)
# Simulamos ventas con estacionalidad (suben y bajan) y un poco de ruido impredecible
ventas = 50 + 20 * np.sin(0.1 * tiempo) + np.random.normal(0, 2, 1000)

# ESCALADO: Las LSTM odian los números grandes. Comprimimos todo entre 0 y 1.
ventas_max = np.max(ventas)
ventas_min = np.min(ventas)
ventas_escaladas = (ventas - ventas_min) / (ventas_max - ventas_min)

# ==========================================
# 2. PREPARACIÓN DE "VENTANAS DE TIEMPO"
# ==========================================
# Cortamos la historia en bloques: Le damos 30 días para que adivine el día 31
X = [] # El pasado (30 días)
y = [] # El futuro (día 31)

for i in range(len(ventas_escaladas) - 30):
    X.append(ventas_escaladas[i:i+30])
    y.append(ventas_escaladas[i+30])

X = np.array(X)
y = np.array(y)

# TensorFlow exige forma 3D para LSTMs: [Lotes, Pasos de Tiempo, Características]
X = X.reshape((X.shape[0], X.shape[1], 1))
print(f"Datos listos. Tenemos {X.shape[0]} bloques de entrenamiento.")
print("-" * 40)

modelo=tf.keras.Sequential([
    tf.keras.layers.Input(shape=(30,1)), # 30 días, 1 variable para predecir
    tf.keras.layers.LSTM(units=64, return_sequences=True), # Capa LSTM con 64 unidades y salida para cada paso de tiempo
    tf.keras.layers.LSTM(units=32), # Segunda capa LSTM con 32
    tf.keras.layers.Dense(units=16, activation='relu'), # Capa oculta con ReLU para procesar la información temporal
    tf.keras.layers.Dense(units=1) # Capa de salida para predecir el valor del día 31 sin activación (regresión) porque queremos un número continuo
])

modelo.compile(optimizer=tf.keras.optimizers.Adam(0.01), loss='mean_squared_error')

modelo.fit(X, y, epochs=50, verbose=0)

# ==========================================
# Tomamos los últimos 30 días exactos de nuestro registro
ultimos_30_dias = ventas_escaladas[-30:]
# Le damos la forma 3D que exige la LSTM
ventana_prediccion = ultimos_30_dias.reshape((1, 30, 1)) # Hacemos la predicción para el día 31, usamos reshape para darle la forma 3D que exige la LSTM
prediccion=modelo.predict(ventana_prediccion)
ventas_mananas=prediccion[0][0] * (ventas_max - ventas_min) + ventas_min # Desescalamos la predicción para obtener el valor real
print("Las ventas estimada para mañana son: ", ventas_mananas)
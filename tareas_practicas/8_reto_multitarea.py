import tensorflow as tf

entrada_auto = tf.keras.layers.Input(shape=(3,))
cerebro_principal = tf.keras.layers.Dense(units=16, activation='relu')(entrada_auto)

salida_precio = tf.keras.layers.Dense(units=1, name='prediccion_precio')(cerebro_principal)

salida_mecanico = tf.keras.layers.Dense(units=1, activation='sigmoid', name='prediccion_mecanico')(cerebro_principal)

modelo = tf.keras.Model(inputs=entrada_auto, outputs=[salida_precio, salida_mecanico])

# Así se compila un modelo de dos cabezas:
modelo.compile(
    optimizer='adam',
    loss={
        'prediccion_precio': 'mean_squared_error',
        'prediccion_mecanico': 'binary_crossentropy'
    }
)

modelo.summary()
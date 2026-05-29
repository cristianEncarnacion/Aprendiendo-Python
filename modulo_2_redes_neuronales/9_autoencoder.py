import tensorflow as tf

entrada_datos= tf.keras.layers.Input(shape=(64,))
encoder=tf.keras.layers.Dense(units=32, activation='relu')(entrada_datos)
capa2=tf.keras.layers.Dense(units=8, activation='relu')(encoder)

decoder=tf.keras.layers.Dense(units=32, activation='relu')(capa2)
salida_datos=tf.keras.layers.Dense(units=64, activation='sigmoid')(decoder)


autoencoder = tf.keras.Model(inputs=entrada_datos, outputs=salida_datos)

autoencoder.summary()
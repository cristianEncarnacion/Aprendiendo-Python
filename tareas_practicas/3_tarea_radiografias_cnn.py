import tensorflow as tf
#Modelo con arquitectura de CNN(Red neuronal convolucional)
modelo=tf.keras.Sequential([
    tf.keras.layers.Input(shape=(150,150,1)), # Especificamos que la entrada tiene 150x150 píxeles y 1 canal (blanco y negro)

    # Capa de convolución
    #Escaner, con un filtro de 32 la cual son las lupas/escaneres diferentes.
    # Kernel_size=(3,3) es el tamaño de nuestra lupa
    tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), activation='relu'), # Capa de convolución con 32 filtros y función de activación ReLU


    #Capa de polling este es para ahorrar memoria.
    tf.keras.layers.MaxPooling2D(pool_size=(2,2)),

     # Capa de convolución
    #Escaner, con un filtro de 32 la cual son las lupas/escaneres diferentes.
    # Kernel_size=(3,3) es el tamaño de nuestra lupa
    tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), activation='relu'), # Capa de convolución con 32 filtros y función de activación ReLU


    #Capa de polling este es para ahorrar memoria.
    tf.keras.layers.MaxPooling2D(pool_size=(2,2)),

    # Capa de aplanamiento
    tf.keras.layers.Flatten(),

    # Razona lo que vio el escaner
    tf.keras.layers.Dense(units=128, activation='relu'), # Capa densamente conectada con 128 neuronas y función de activación ReLU

    tf.keras.layers.Dense(units=1, activation='sigmoid') # Capa de salida con 1 neurona y función de activación Sigmoid
    
    
])

modelo.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss="binary_crossentropy")

modelo.summary()
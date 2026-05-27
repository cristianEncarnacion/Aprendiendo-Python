from sklearn.neighbors import KNeighborsClassifier

etiquetas_entrenamiento=[0,1,1,0]

# Cada sub-lista representa una fruta con su [Color, Forma]
caracteristicas_entrenamiento = [
    [0, 0],  # Fruta 1
    [1, 1],  # Fruta 2
    [1, 1],  # Fruta 3
    [0, 0]   # Fruta 4
]

modelo=KNeighborsClassifier(n_neighbors=3)

modelo.fit(caracteristicas_entrenamiento, etiquetas_entrenamiento)
# Definimos los datos de la nueva fruta en una matriz: [Color Rojo, Forma Circular]
fruta_misteriosa = [[0, 1]]

# Le pedimos al modelo entrenado que haga su predicción
resultado = modelo.predict(fruta_misteriosa)

# Mostramos el resultado en pantalla
print(resultado)
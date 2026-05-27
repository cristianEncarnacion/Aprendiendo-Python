from sklearn.neighbors import KNeighborsClassifier

etiquetas_entrenamiento=[0,1,0,1] # 0 resfriado, 1 estornudo

caracteristicas_entrenamiento = [
    [1, 1],  # Paciente 1: con fiebre (1) (1) -> resfriado (0)
    [0, 1],   # Paciente 2: sin fiebre (0) (1) -> estornudo (1)
    [1,0], # Paciente 3: con fiebre (1), si (0) -> resfriado (0)
    [0,1]    # Paciente 4: sin fiebre (0) (1) -> estornudo (1)
]

modelo=KNeighborsClassifier(n_neighbors=3) # Creamos un modelo KNN con k=3, lo que significa que el modelo considerará los 3 vecinos más cercanos para hacer una predicción.
modelo.fit(caracteristicas_entrenamiento,etiquetas_entrenamiento) # Entrenamos el modelo con los datos de características y etiquetas de entrenamiento.

nuevo_paciente=[[0,1]] # Nuevo paciente: sin fiebre (0), con tos (1)

resultado=modelo.predict(nuevo_paciente) # Le pedimos al modelo entrenado que haga su predicción para el nuevo paciente.

print(resultado) # Mostramos el resultado en pantalla. El resultado será un array que contiene la etiqueta predicha para el nuevo paciente, donde 0 representa resfriado y 1 representa estornudo.
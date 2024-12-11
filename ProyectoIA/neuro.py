import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# Definición del laberinto
maze = np.array([
    [1, 1, 1, 1, 1],
    [1, 2, 0, 0, 1],
    [1, 0, 1, 0, 1],
    [1, 0, 0, 3, 1],
    [1, 1, 1, 1, 1],
])

# Generar datos de entrenamiento
def generate_training_data(maze):
    """
    Genera datos de entrenamiento basados en el laberinto.
    Cada posición transitable tiene acciones válidas hacia posiciones adyacentes.
    """
    data = []
    labels = []
    for i in range(maze.shape[0]):
        for j in range(maze.shape[1]):
            if maze[i, j] == 0 or maze[i, j] == 2:  # Solo pasillos o inicio
                for action, (di, dj) in enumerate([(0, -1), (0, 1), (-1, 0), (1, 0)]):  # Izquierda, Derecha, Arriba, Abajo
                    ni, nj = i + di, j + dj
                    if 0 <= ni < maze.shape[0] and 0 <= nj < maze.shape[1] and maze[ni, nj] != 1:
                        data.append([i, j])
                        labels.append(action)
    return np.array(data), np.array(labels)

data, labels = generate_training_data(maze)

# Normalizar datos
scaler = StandardScaler()
data = scaler.fit_transform(data)

# Dividir en entrenamiento y validación
x_train, x_val, y_train, y_val = train_test_split(data, labels, test_size=0.2, random_state=42)

# Crear modelo
model = models.Sequential([
    layers.Dense(128, activation='relu', input_shape=(2,)),
    layers.Dense(128, activation='relu'),
    layers.Dense(4, activation='softmax')  # Salida: probabilidad de cada acción
])

# Compilar modelo
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Entrenar modelo
model.fit(x_train, y_train, epochs=50, validation_data=(x_val, y_val))

# Guardar modelo para su uso posterior
model.save("maze_solver_improved.keras")

# Lógica de resolución del laberinto con retroceso y registro de la ruta
def solve_maze(model, maze, start, goal):
    """
    Resuelve el laberinto usando el modelo entrenado y registra la ruta completa.
    """
    scaler = StandardScaler()
    scaler.fit_transform(np.argwhere((maze == 0) | (maze == 2) | (maze == 3)))  # Escalar posiciones transitables

    current_position = start
    path = [current_position]
    visited = set()

    while current_position != goal:
        visited.add(tuple(current_position))
        state = scaler.transform([current_position])
        action_probabilities = model.predict(state, verbose=0)
        action = np.argmax(action_probabilities)

        di, dj = [(0, -1), (0, 1), (-1, 0), (1, 0)][action]  # Acción a coordenadas
        next_position = [current_position[0] + di, current_position[1] + dj]

        if tuple(next_position) in visited or maze[next_position[0], next_position[1]] == 1:
            # Si el siguiente movimiento es inválido o ya visitado, retrocede
            path.pop()
            if not path:
                print("No hay solución disponible.")
                return []
            current_position = path[-1]
        else:
            current_position = next_position
            path.append(current_position)

    return path

# Función para mostrar el recorrido paso a paso
def visualize_path(maze, path):
    """
    Visualiza el recorrido del ratón paso a paso en el laberinto.
    """
    plt.ion()  # Modo interactivo
    for step in path:
        maze_copy = maze.copy()
        maze_copy[step[0], step[1]] = 4  # Representar la posición actual del ratón
        plt.imshow(maze_copy, cmap="cool")
        plt.xticks([])
        plt.yticks([])
        plt.title("Recorrido del Ratón en el Laberinto")
        plt.pause(0.5)  # Pausa para simular el movimiento

    plt.ioff()
    plt.show()

# Ejemplo de uso
start = [1, 1]
goal = [3, 3]
path = solve_maze(model, maze, start, goal)
if path:
    print("Ruta encontrada:", path)
    visualize_path(maze, path)

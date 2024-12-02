import numpy as np
import random

# Definir el laberinto (0 = libre, 1 = pared)
laberinto = np.array([
    [0, 1, 0, 0, 0],
    [0, 1, 0, 1, 0],
    [0, 0, 0, 1, 0],
    [0, 1, 0, 0, 0],
    [0, 0, 0, 1, 0]
])

# Establecer las posiciones de inicio y meta
start = (0, 0)
goal = (4, 4)

class DNA:
    def __init__(self, num_individuos, num_seleccion, num_generacion, tasa_mutacion):
        self.num_individuos = num_individuos
        self.num_seleccion = num_seleccion
        self.num_generacion = num_generacion
        self.tasa_mutacion = tasa_mutacion  # Renombrado de "mutacion" a "tasa_mutacion"
        self.direccion = ['arriba', 'abajo', 'izquierda', 'derecha']  # Movimientos posibles

    def crear_raton(self):
        # El ratón se representa como una lista de movimientos, donde cada movimiento es aleatorio
        return [random.choice(self.direccion) for _ in range(50)]  # 50 movimientos como ejemplo

    def crear_poblacion(self):
        return [self.crear_raton() for _ in range(self.num_individuos)]

    def fitness(self, raton):
        x, y = start
        for movimiento in raton:
            if movimiento == 'arriba' and x > 0 and laberinto[x-1][y] == 0:
                x -= 1
            elif movimiento == 'abajo' and x < len(laberinto)-1 and laberinto[x+1][y] == 0:
                x += 1
            elif movimiento == 'izquierda' and y > 0 and laberinto[x][y-1] == 0:
                y -= 1
            elif movimiento == 'derecha' and y < len(laberinto[0])-1 and laberinto[x][y+1] == 0:
                y += 1

            # Si llega a la meta, la evaluación es máxima
            if (x, y) == goal:
                return 100

        # Retornar la distancia del ratón a la meta
        return - (abs(x - goal[0]) + abs(y - goal[1]))

    def seleccion(self, poblacion):
        puntuacion = [(self.fitness(raton), raton) for raton in poblacion]
        puntuacion.sort(reverse=True, key=lambda x: x[0])
        return [r[1] for r in puntuacion[:self.num_seleccion]]

    def reproduccion(self, seleccion):
        poblacion = []
        for _ in range(self.num_individuos):
            padre1, padre2 = random.sample(seleccion, 2)
            punto = random.randint(1, len(padre1) - 1)
            hijo = padre1[:punto] + padre2[punto:]
            poblacion.append(hijo)
        return poblacion

    def mutacion(self, poblacion):
        for i in range(len(poblacion)):
            if random.random() < self.tasa_mutacion:  # Se usa tasa_mutacion en lugar de mutacion
                punto = random.randint(0, len(poblacion[i]) - 1)
                poblacion[i][punto] = random.choice(self.direccion)
        return poblacion

    def mostrar_laberinto(self, x, y):
        # Crea una copia del laberinto y marca la posición del ratón
        laberinto_temp = laberinto.copy()
        laberinto_temp[x][y] = 2  # Marca al ratón con un 2
        for row in laberinto_temp:
            print(' '.join(str(cell) for cell in row))
        print()  # Línea vacía para separar las representaciones del laberinto

    def run_genetico(self):
        poblacion = self.crear_poblacion()
        for generacion in range(self.num_generacion):
            print(f"Generación: {generacion}")
            seleccionados = self.seleccion(poblacion)
            poblacion = self.reproduccion(seleccionados)
            poblacion = self.mutacion(poblacion)

            # Encontrar el mejor ratón de la generación
            mejores_raton = max(poblacion, key=self.fitness)
            print("Mejor ratón de esta generación: ", mejores_raton)
            print("Fitness: ", self.fitness(mejores_raton))

            # Simulación del recorrido del ratón
            x, y = start
            for movimiento in mejores_raton:
                if movimiento == 'arriba' and x > 0 and laberinto[x-1][y] == 0:
                    x -= 1
                elif movimiento == 'abajo' and x < len(laberinto)-1 and laberinto[x+1][y] == 0:
                    x += 1
                elif movimiento == 'izquierda' and y > 0 and laberinto[x][y-1] == 0:
                    y -= 1
                elif movimiento == 'derecha' and y < len(laberinto[0])-1 and laberinto[x][y+1] == 0:
                    y += 1

                # Mostrar el laberinto en cada paso
                self.mostrar_laberinto(x, y)

                # Si llega a la meta, detener el recorrido
                if (x, y) == goal:
                    print("¡El ratón ha llegado a la meta!")
                    break

def main():
    # Inicializar el algoritmo genético
    model = DNA(num_individuos=10, num_seleccion=5, num_generacion=50, tasa_mutacion=0.05)
    
    # Ejecutar el algoritmo genético
    model.run_genetico()

if __name__ == '__main__':
    main()

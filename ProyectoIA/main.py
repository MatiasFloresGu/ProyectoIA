import numpy as np
import random
import matplotlib.pyplot as plt



maze = [
    [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
    [1,2,0,0,1,0,0,0,0,0,0,1,0,0,0,1,0,0,0,1],
    [1,1,1,0,1,0,1,1,1,1,0,1,1,1,0,1,0,1,1,1],
    [1,0,0,0,1,0,1,0,0,1,0,0,0,0,0,1,0,0,0,1],
    [1,0,1,1,1,0,1,0,1,1,1,1,1,1,1,1,1,1,0,1],
    [1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,1],
    [1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,1,0,1,0,1],
    [1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,1,0,1],
    [1,0,1,0,1,1,1,1,1,1,1,1,1,1,0,1,0,1,0,1],
    [1,0,1,0,0,0,0,0,0,0,0,0,0,1,0,1,0,1,0,1],
    [1,0,1,1,1,1,1,1,1,1,1,1,0,1,0,1,0,1,0,1],
    [1,0,0,0,0,0,0,0,0,0,0,1,0,1,0,1,0,1,0,1],
    [1,1,1,1,1,1,1,1,1,1,0,1,0,1,0,1,0,1,0,1],
    [1,0,0,0,0,0,0,0,0,1,0,1,0,0,0,1,0,1,0,1],
    [1,0,1,1,1,1,1,1,0,1,0,1,1,1,1,1,0,1,0,1],
    [1,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,1],
    [1,0,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1],
    [1,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
    [1,0,0,0,1,0,1,1,1,1,1,1,1,1,1,1,1,1,0,1],
    [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
]


start = (1,1)
goal = (18,18)

maze_array = np.array(maze)


plt.figure(figsize=(10,10))
plt.imshow(maze_array, cmap="binary") 
plt.title("Visualización del Laberinto")
plt.xticks([])
plt.yticks([])  
plt.grid(visible=False)  
plt.show()


class DNA:
    def __init__(self, num_individuos, num_seleccion, num_generacion, tasa_mutacion):
        self.num_individuos = num_individuos
        self.num_seleccion = num_seleccion
        self.num_generacion = num_generacion
        self.tasa_mutacion = tasa_mutacion 
        self.direccion = ['arriba', 'abajo', 'izquierda', 'derecha']  

    # El ratón se representa como una lista de movimientos, donde cada movimiento es aleatorio
    def crear_raton(self):
        
        return [random.choice(self.direccion) for _ in range(50)]  

    def crear_poblacion(self):
        return [self.crear_raton() for _ in range(self.num_individuos)]

    # Retornar la distancia del ratón a la meta (no optimo)
    def fitness(self, raton):
        x, y = start
        for movimiento in raton:
            if movimiento == 'arriba' and x > 0 and maze[x-1][y] == 0:
                x -= 1
            elif movimiento == 'abajo' and x < len(maze)-1 and maze[x+1][y] == 0:
                x += 1
            elif movimiento == 'izquierda' and y > 0 and maze[x][y-1] == 0:
                y -= 1
            elif movimiento == 'derecha' and y < len(maze[0])-1 and maze[x][y+1] == 0:
                y += 1

            # Si llega a la meta, la evaluación es máxima
            if (x, y) == goal:
                return 100

        return - (abs(x - goal[0]) + abs(y - goal[1]))

    #Recibe una poblacion en la cual se optiene la poblacion para despues ordenarla y seleccionar en base al numero de la seleccion
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
            if random.random() < self.tasa_mutacion:  
                punto = random.randint(0, len(poblacion[i]) - 1)
                poblacion[i][punto] = random.choice(self.direccion)
        return poblacion

    def mostrar_laberinto(self, x, y):
        maze_copy = maze_array.copy()
        maze_copy[x, y] = 2 

        plt.imshow(maze_copy, cmap="cool")
        plt.xticks([])
        plt.yticks([])
        plt.grid(visible=False)
        plt.pause(0.1)  


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
                if movimiento == 'arriba' and x > 0 and maze[x-1][y] == 0:
                    x -= 1
                elif movimiento == 'abajo' and x < len(maze)-1 and maze[x+1][y] == 0:
                    x += 1
                elif movimiento == 'izquierda' and y > 0 and maze[x][y-1] == 0:
                    y -= 1
                elif movimiento == 'derecha' and y < len(maze[0])-1 and maze[x][y+1] == 0:
                    y += 1

                
                self.mostrar_laberinto(x, y)
                if (x, y) == goal:
                    print("¡El ratón ha llegado a la meta!")
                    plt.ioff()
                    plt.show()
                    break

def main():
    model = DNA(num_individuos=10, num_seleccion=5, num_generacion=50, tasa_mutacion=0.05)
    
    model.run_genetico()

if __name__ == '__main__':
    main()
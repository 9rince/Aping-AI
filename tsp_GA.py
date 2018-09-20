import numpy as np
import random
import matplotlib.pyplot as plt
import networkx as nx


class Travelling_salesman:
    def __init__(self,no_of_cities = 5):
        self.no_of_cities = no_of_cities
        self.city_cordinates = []
        self.cities = []
        self.solutions = []
        self.distance = []
        self.no_of_offsprings = 100
        self.no_of_cycles = 1000
        self.parent_1 = []
        self.parent_2 = []
        self.bst_dist = float('inf')
        self.bst_parent = []

    def populate_cities(self):
        for i in range(self.no_of_cities):
            self.city_cordinates.append([random.uniform(-10.,10.),random.uniform(-10.,10.)])
            self.cities.append(i)

    def populate_solution(self):
        x = self.cities
        for i in range (self.no_of_offsprings):
            random.shuffle(x)
            self.solutions.append(list(x))

    def distanz(self,city_a,city_b):
        return np.linalg.norm(np.array(self.city_cordinates[city_a])-np.array(self.city_cordinates[city_b]))

    def crossover(self,p1,p2):
        cut_at = np.random.randint(len(p1))
        offspring = p1[:cut_at]
        for i in p2:
            if i not in offspring:
                offspring = offspring + [i]
        return offspring

    def mutate(self,offspring):
        swap = np.random.randint(0,len(offspring),size=(2))
        offspring[swap[0]],offspring[swap[1]] = offspring[swap[1]],offspring[swap[0]]
        return offspring

    def calculate_distance(self):
        self.distance = []
        for i in range(self.no_of_offsprings):
            dist = 0.
            for j in range(1,self.no_of_cities):
                # print(j,len(self.solutions[i]))
                dist += self.distanz(self.solutions[i][j-1],self.solutions[i][j])
            dist += self.distanz(self.solutions[i][0],self.solutions[i][-1])
            self.distance.append(dist)




    def stochastic_universal_selection(self):
        norm_fitnz = 1 - (np.array(self.distance)/np.sum(np.array(self.distance)))
        norm_fitnz = norm_fitnz/np.sum(norm_fitnz)
        a,b  =  np.random.uniform(0,1),np.random.uniform(0,1)
        p1 = []
        p2 = []
        for i in range(1,self.no_of_offsprings):
            norm_fitnz[i] = norm_fitnz[i-1]+norm_fitnz[i]
            if (a<=norm_fitnz[i] and a>=norm_fitnz[i-1]) :
                p1 = list(self.solutions[i])
                if self.bst_dist > self.distance[i]:
                    self.bst_dist = self.distance[i]
                    self.bst_parent = p1
                    print(self.distance[i],min(self.distance))
            if (b<=norm_fitnz[i] and b>=norm_fitnz[i-1]) :
                p2 = list(self.solutions[i])
                if self.bst_dist > self.distance[i]:
                    self.bst_dist = self.distance[i]
                    self.bst_parent = p2
                    print(self.distance[i],min(self.distance))
        if p1 and p2:
            self.parent_1 = p1
            self.parent_2 = p2
        else:
            self.stochastic_universal_selection()

    def reproduce(self):
        self.solutions = []
        i = self.no_of_offsprings
        while i>0:
            p1,p2 = self.parent_1,self.parent_2
            offspring = self.crossover(p1,p2)
            offspring = self.mutate(offspring)
            offspring = self.mutate(offspring)
            self.solutions.append(offspring)
            i -= 1

    def plot_cities(self):
        G = nx.DiGraph()
        x = [str(self.bst_parent[i]) for i in range(self.no_of_cities)]
        pos = {str(self.bst_parent[i]):self.city_cordinates[self.bst_parent[i]] for i in range(self.no_of_cities)}
        G.add_nodes_from(x)
        for i in range(len(x)):
            G.add_edge(x[i-1],x[i])
        G.add_edge(x[-1],x[0])
        # print(G.nodes())
        # print(G.edges())
        nx.draw(G , pos , with_labels = True)
        plt.show()

    def evolve(self):
        self.populate_cities()
        self.populate_solution()
        k = self.no_of_cycles
        while(k > 0):
            self.calculate_distance()
            self.stochastic_universal_selection() # Selecting Parents
            self.reproduce()
            print(k,min(self.distance))
            k -= 1


Rohan = Travelling_salesman(no_of_cities=100)
Rohan.evolve()
print(Rohan.bst_dist,Rohan.bst_parent)
Rohan.plot_cities()

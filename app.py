import random
import operator
import database
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from city import City
from fitness import Fitness
from database import Database

SELECTION_METHOD = 'TOURNAMENT' #ROULETTE or TOURNAMENT
CITIES = Database.get_cities()

'''
  Generate a random individual
'''
def createRoute(cities):
    route = random.sample(cities, len(cities))
    return route

'''
  Generate a population with swapped individuals
'''
def initialPopulation(popSize, cities):
    population = []

    for i in range(0, popSize):
        population.append(createRoute(cities))
    return population

def rankRoutes(population):
    fitnessResults = {}
    for i in range(0,len(population)):
        fitnessResults[i] = Fitness(population[i]).routeFitness()
    return sorted(fitnessResults.items(), key = operator.itemgetter(1), reverse = True)

def selection(popRanked, eliteSize, method):
    results = []
    df = pd.DataFrame(np.array(popRanked), columns=["Index","Fitness"])

    #Cumulative Sum from Fitness column
    df['cumulative_sum'] = df.Fitness.cumsum()

    #Cumulative Percentage
    df['cumulative_percentage'] = 100*df.cumulative_sum/df.Fitness.sum()

    #Select elite
    for i in range(0, eliteSize):
        results.append(popRanked[i][0])

    if (method == 'TOURNAMENT'):
      for i in range(0, len(popRanked) - eliteSize):
        in_tournament = random.sample(popRanked, 20)
        results.append(in_tournament[0][0])

    if (method == 'ROULETTE'):
      for i in range(0, len(popRanked) - eliteSize):
          pick = 100*random.random()
          for i in range(0, len(popRanked)):
              cumulative_percentage = df.iat[i,3]

              if pick <= cumulative_percentage:
                  results.append(popRanked[i][0])
                  break
    
    return results

def matingPool(population, selectionResults):
    matingpool = []
    for i in range(0, len(selectionResults)):
        index = selectionResults[i]
        matingpool.append(population[index])

    return matingpool


'''
 Crossover Offspring technique for individual
'''
def breed(parent1, parent2):
    child = []
    childP1 = []
    childP2 = []
    
    geneA = int(random.random() * len(parent1))
    geneB = int(random.random() * len(parent1))

    startGene = min(geneA, geneB)
    endGene = max(geneA, geneB)

    for i in range(startGene, endGene):
        childP1.append(parent1[i])
        
    childP2 = [item for item in parent2 if item not in childP1]

    child = childP1 + childP2
    return child

'''
 Apply Crossover Offspring technique for all mating pool
'''
def breedPopulation(matingpool, eliteSize):
    children = []
    length = len(matingpool) - eliteSize
    pool = random.sample(matingpool, len(matingpool))

    for i in range(0,eliteSize):
        children.append(matingpool[i])
    
    for i in range(0, length):
        child = breed(pool[i], pool[len(matingpool)-i-1])
        children.append(child)
    return children

def mutate(individual, mutationRate):
    for swapped in range(len(individual)):
        if(random.random() < mutationRate):
            swapWith = int(random.random() * len(individual))
            
            city1 = individual[swapped]
            city2 = individual[swapWith]

            individual[swapped] = city2
            individual[swapWith] = city1
    return individual

def mutatePopulation(population, mutationRate):
    mutatedPop = []

    for ind in range(0, len(population)):
        mutatedInd = mutate(population[ind], mutationRate)
        mutatedPop.append(mutatedInd)
    return mutatedPop

def nextGeneration(currentGen, eliteSize, mutationRate, method):
    popRanked = rankRoutes(currentGen)
    selectionResults = selection(popRanked, eliteSize, method)
    matingpool = matingPool(currentGen, selectionResults)
    children = breedPopulation(matingpool, eliteSize)
    nextGeneration = mutatePopulation(children, mutationRate)
    return nextGeneration

def run(population, popSize, eliteSize, mutationRate, generations, method, showPlot):
    all_progress = []
    pop = initialPopulation(popSize, population)
    progress = rankRoutes(pop)[0][1]
    all_progress.append(1 / progress)
    print("Initial distance: " + str(1 / progress))
    
    for i in range(0, generations):
        pop = nextGeneration(pop, eliteSize, mutationRate, method)
        progress = rankRoutes(pop)[0][1]
        all_progress.append(1 / progress)

    
    print("Final distance: " + str(1 / progress))

    if (showPlot):
      plt.plot(all_progress)
      plt.ylabel('Distância')
      plt.xlabel('Geração')
      plt.show()

run(
  population=CITIES, 
  popSize=100, 
  eliteSize=20, 
  mutationRate=0.0009,
  generations=500, 
  method=SELECTION_METHOD,
  showPlot=True
)
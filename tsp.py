import random
import math
import time
from multiprocessing import Pool, cpu_count

def load_cities(file_path):
    cities = []
    with open(file_path, 'r') as f:
        for line in f:
            node_city_val = line.split()
            cities.append([node_city_val[0], float(node_city_val[1]), float(node_city_val[2])])
    return cities

def calculate_distance(cities):
    total_distance = 0
    for i in range(len(cities) - 1):
        cityA = cities[i]
        cityB = cities[i + 1]
        d = math.sqrt((cityB[1] - cityA[1]) ** 2 + (cityB[2] - cityA[2]) ** 2)
        total_distance += d
    cityA = cities[0]
    cityB = cities[-1]
    d = math.sqrt((cityB[1] - cityA[1]) ** 2 + (cityB[2] - cityA[2]) ** 2)
    total_distance += d
    return total_distance

def initialize_population(cities, size):
    population = []
    for _ in range(size):
        c = cities.copy()
        random.shuffle(c)
        distance = calculate_distance(c)
        population.append([distance, c])
    return population

def genetic_algorithm(population, len_cities, tournament_selection_size, mutation_rate, crossover_rate, target, max_generations):
    gen_number = 0
    best_distances = []
    best_distance = float('inf')
    while gen_number < max_generations:
        new_population = []
        new_population.append(sorted(population)[0])
        new_population.append(sorted(population)[1])
        for _ in range(int((len(population) - 2) / 2)):
            if random.random() < crossover_rate:
                parent_chromosome1 = sorted(random.sample(population, tournament_selection_size))[0]
                parent_chromosome2 = sorted(random.sample(population, tournament_selection_size))[0]
                point = random.randint(0, len_cities - 1)
                child_chromosome1 = parent_chromosome1[1][0:point]
                for j in parent_chromosome2[1]:
                    if j not in child_chromosome1:
                        child_chromosome1.append(j)
                child_chromosome2 = parent_chromosome2[1][0:point]
                for j in parent_chromosome1[1]:
                    if j not in child_chromosome2:
                        child_chromosome2.append(j)
                    else:
                        child_chromosome1 = random.choice(population)[1]
                        child_chromosome2 = random.choice(population)[1]
            if random.random() < mutation_rate:
                point1 = random.randint(0, len_cities - 1)
                point2 = random.randint(0, len_cities - 1)
                child_chromosome1[point1], child_chromosome1[point2] = child_chromosome1[point2], child_chromosome1[point1]
                point1 = random.randint(0, len_cities - 1)
                point2 = random.randint(0, len_cities - 1)
                child_chromosome2[point1], child_chromosome2[point2] = child_chromosome2[point2], child_chromosome2[point1]
            new_population.append([calculate_distance(child_chromosome1), child_chromosome1])
            new_population.append([calculate_distance(child_chromosome2), child_chromosome2])
        population = new_population
        gen_number += 1
        if gen_number % 10 == 0:
            best_distance = sorted(population)[0][0]
            print(f"Generation: {gen_number}, Best Distance: {best_distance}")
            best_distances.append(best_distance)
            if best_distance < target:
                print("Target distance reached!")
                break
    answer = sorted(population)[0]
    return answer, gen_number, best_distances

def main():
    POPULATION_SIZE = 1000
    TOURNAMENT_SELECTION_SIZE = 4
    MUTATION_RATE = 0.1
    CROSSOVER_RATE = 0.8
    TARGET_DISTANCE = 450.0
    MAX_GENERATIONS = 100
    cities = load_cities(r"C:\Users\tanuj\Downloads\TSP51.txt")
    
    # Serial execution
    start_time_serial = time.time()
    population_serial = initialize_population(cities, POPULATION_SIZE)
    best_solution_serial, generations_serial, best_distances_serial = genetic_algorithm(
        population_serial, len(cities), TOURNAMENT_SELECTION_SIZE, MUTATION_RATE,
        CROSSOVER_RATE, TARGET_DISTANCE, MAX_GENERATIONS
    )
    end_time_serial = time.time()
    print("Serial Execution Time:", end_time_serial - start_time_serial)
    
    # Parallel execution
    start_time_parallel = time.time()
    with Pool(processes=cpu_count()) as pool:
        population_parallel = initialize_population(cities, POPULATION_SIZE)
        results_parallel = pool.starmap(genetic_algorithm, [
            (population_parallel, len(cities), TOURNAMENT_SELECTION_SIZE, MUTATION_RATE,
             CROSSOVER_RATE, TARGET_DISTANCE, MAX_GENERATIONS)
        ])
    end_time_parallel = time.time()
    print("Parallel Execution Time:", end_time_parallel - start_time_parallel)
    
    best_solution_parallel = sorted(results_parallel)[0]
    best_distance_parallel = best_solution_parallel[0][0]
    if best_solution_serial[0] < best_distance_parallel:
        best_solution = best_solution_serial
        generations = generations_serial
        best_distances = best_distances_serial
        execution_time = end_time_serial - start_time_serial
    else:
        best_solution = best_solution_parallel[0]
        generations = best_solution[1]
        best_distances = best_solution[2]
        execution_time = end_time_parallel - start_time_parallel
    print("Best Distances:", best_distances)
    print("\n----------------------------------------------------------------")
    print(f"Generation: {generations}")
    print(f"Fittest chromosome distance after training: {best_solution[0]}")
    print(f"Target distance: {TARGET_DISTANCE}")
    print(f"Execution Time: {execution_time}")
    print("----------------------------------------------------------------\n")

if __name__ == "__main__":
    main()

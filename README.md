* **Project:** A basic 2D simulation demonstrating artificial evolution using a genetic algorithm.
* **Entities:** Simple "agents" or "critters".
* **Environment:** A bounded 2D space (e.g., a square grid or continuous area) containing resources (e.g., "food") and possibly obstacles or hazards.
* **Agent Behavior:** Agents can perceive their immediate surroundings (e.g., nearby food, walls), move, consume resources, and expend energy. Their actions are determined by a simple AI controller (like a neural network).
* **Evolution Mechanism:** A Genetic Algorithm (GA) will evolve the parameters (genes) of the agents' AI controllers.
    * **Fitness:** Measured by survival time, resources gathered, distance traveled, or a combination.
    * **Selection:** Agents with higher fitness are more likely to reproduce.
    * **Reproduction:** Offspring inherit (and potentially combine) genes from parent(s).
    * **Mutation:** Small random changes are introduced into the offspring's genes.
* **Goal:** Observe how agent behaviors adapt over generations to better survive and thrive in the environment.
* **Technology:** Python (using libraries like `NumPy` for calculations and potentially `Pygame` or `Matplotlib` for simple visualization).
* **Timeline:** Focus on understanding the core components and getting a *very basic* structural implementation within the hour timeframe. Visualization might be rudimentary or text-based initially.

**Core Components You Need to Think About**

1.  **The Environment:**
    * Dimensions (width, height).
    * How are resources (food) distributed? Do they respawn?
    * Are there walls or obstacles?
    * How is time simulated (discrete steps/ticks)?

2.  **The Agent:**
    * **State:** Position (x, y), energy level, orientation/direction.
    * **Sensors:** What can it perceive? (e.g., distance/direction to nearest food, distance to wall in front). These become inputs to its AI.
    * **Actuators:** What can it do? (e.g., move forward, turn left/right, eat). These are the outputs of its AI.
    * **Genes:** A representation of the parameters controlling its behavior (e.g., the weights and biases of a neural network). Usually represented as a list or array of numbers.
    * **Energy Dynamics:** Moving costs energy, eating gains energy. If energy drops to zero, the agent "dies".

3.  **The AI Controller (Agent's "Brain"):**
    * A simple Feedforward Neural Network is a common choice.
    * **Inputs:** Sensor readings.
    * **Outputs:** Action probabilities or direct action commands (e.g., thrust, turn angle).
    * **Weights & Biases:** These are the values determined by the agent's "genes" and are what the genetic algorithm evolves.

4.  **The Genetic Algorithm (GA):**
    * **Population:** A collection of agents.
    * **Evaluation:** Run the simulation for a set duration (or until all agents die). Calculate the fitness score for each agent based on its performance (e.g., survival time + food eaten).
    * **Selection:** Choose agents to be "parents" for the next generation. Higher fitness increases the chance of being selected (e.g., Tournament Selection, Roulette Wheel Selection).
    * **Crossover:** Combine the genes (neural network weights) of two parents to create offspring genes. (e.g., single-point crossover, uniform crossover).
    * **Mutation:** Apply small, random changes to the offspring's genes. This maintains genetic diversity and allows for new traits to emerge. (e.g., adding small random values to weights).
    * **Replacement:** Create the new population for the next generation, often replacing the old one entirely or keeping a few top performers (elitism).

5.  **The Simulation Loop:**
    * Initialize the environment and a random population of agents.
    * **For each generation:**
        * Run the simulation step-by-step for a fixed duration:
            * **For each agent:**
                * Get sensor readings from the environment.
                * Feed sensor readings into the agent's neural network.
                * Get action outputs from the network.
                * Apply actions (move, turn), update agent state.
                * Handle interactions (eating food, hitting walls).
                * Update energy levels.
                * Check for death conditions.
            * Update the environment (e.g., respawn food).
            * (Optional) Render the current state for visualization.
        * Evaluate the fitness of all agents that participated.
        * Perform GA steps (Selection, Crossover, Mutation) to create the next generation's population.
        * Report statistics (e.g., average fitness, best fitness).

**Python Structure & Simplified Code Snippets**

Let's outline this using Python with NumPy. Visualization is skipped here for brevity and to focus on core logic within the time constraint.

```python
import numpy as np
import random

# --- Configuration ---
POPULATION_SIZE = 50
GENOME_LENGTH = 20 # Placeholder: Depends on NN size
MUTATION_RATE = 0.05
MUTATION_STRENGTH = 0.1
SIMULATION_STEPS = 500
ENV_WIDTH = 100
ENV_HEIGHT = 100
NUM_FOOD = 30

# --- Simple Neural Network (Example) ---
# For simplicity, let's assume a network structure is defined
# Its weights are represented by the agent's genome
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class SimpleNN:
    def __init__(self, genome):
        # Example: Assume genome directly encodes weights for a simple structure
        # Needs proper design based on inputs/outputs/layers
        self.weights = np.array(genome)
        # This part needs significant expansion for a real NN!
        # E.g., defining layers, biases, connections based on genome.
        # For this skeleton, we'll pretend the genome *is* the weight matrix.

    def predict(self, inputs):
        # Extremely simplified example: linear combination + activation
        # A real NN would have matrix multiplications, biases, multiple layers
        if self.weights.size == 0 or inputs.size == 0:
             return np.array([0.0, 0.0]) # Default action if NN is invalid

        # Placeholder: Assume inputs match some part of the weights structure
        # This calculation is purely illustrative and needs proper NN math
        try:
            # Example: Assume inputs size matches expected input layer size
            # And weights somehow map inputs to 2 outputs (e.g., move, turn)
            # THIS IS A HUGE SIMPLIFICATION!
            num_inputs = inputs.shape[0]
            # Ensure weights can be reshaped (simplistic assumption)
            expected_shape = (2, num_inputs) # Example: 2 outputs
            if self.weights.size == np.prod(expected_shape):
                 reshaped_weights = self.weights.reshape(expected_shape)
                 output = sigmoid(np.dot(reshaped_weights, inputs))
                 return output # e.g., [move_output, turn_output]
            else:
                 # If genome doesn't match expected NN structure, return default
                 # print(f"Weight size mismatch: {self.weights.size} vs {np.prod(expected_shape)}") # Debugging
                 return np.array([0.0, 0.0])
        except Exception as e:
            # print(f"NN prediction error: {e}") # Debugging
            return np.array([0.0, 0.0]) # Default action on error


# --- Agent Class ---
class Agent:
    def __init__(self, genome=None):
        self.x = random.uniform(0, ENV_WIDTH)
        self.y = random.uniform(0, ENV_HEIGHT)
        self.energy = 100.0
        self.alive = True
        self.fitness = 0.0
        self.angle = random.uniform(0, 2 * np.pi) # Direction facing

        if genome is None:
            # Initialize with random genes if none provided
            self.genome = np.random.uniform(-1, 1, GENOME_LENGTH)
        else:
            self.genome = genome

        self.nn = SimpleNN(self.genome)

    def sense(self, food_positions):
        # Basic sensor: direction/distance to nearest food (simplified)
        # Needs implementation: find nearest food, calculate vector, etc.
        # Also needs: sense walls? sense other agents?
        # Placeholder: returns fixed dummy sensor inputs
        inputs = np.random.rand(5) # Example: 5 sensor inputs
        # A real implementation would calculate these based on self.x, self.y, self.angle and food_positions
        # E.g., inputs = [dist_to_food, angle_to_food, dist_to_wall_fwd, ...]
        # Ensure the number of inputs matches what the NN expects!
        # Adjust GENOME_LENGTH based on actual NN input/output/hidden layers
        global GENOME_LENGTH
        GENOME_LENGTH = 5 * 2 # Example: 5 inputs, 2 outputs, no hidden layer weights
        if len(self.genome) != GENOME_LENGTH:
             self.genome = np.random.uniform(-1, 1, GENOME_LENGTH) # Resize/reinit genome if needed
             self.nn = SimpleNN(self.genome) # Recreate NN
        return inputs[:self.nn.weights.shape[1]] # Ensure input size matches NN expectation

    def act(self, nn_outputs):
        # Interpret NN outputs to change state
        # Example: output[0] = move force, output[1] = turn force
        move_force = nn_outputs[0]
        turn_force = (nn_outputs[1] - 0.5) * 2 # Map [0,1] -> [-1, 1]

        # Update angle and position (simple physics)
        self.angle += turn_force * 0.1 # Max turn rate
        self.x += np.cos(self.angle) * move_force * 1.0 # Max speed
        self.y += np.sin(self.angle) * move_force * 1.0
        self.energy -= 0.1 + move_force * 0.05 # Base energy cost + movement cost

        # Boundary checks
        self.x = np.clip(self.x, 0, ENV_WIDTH)
        self.y = np.clip(self.y, 0, ENV_HEIGHT)

    def update(self, food_positions):
        if not self.alive:
            return

        sensor_inputs = self.sense(food_positions)
        actions = self.nn.predict(sensor_inputs)
        self.act(actions)

        # Check for death
        if self.energy <= 0:
            self.alive = False
            # Fitness calculation happens *after* simulation step/run

        # Check for eating (simplified) - Needs food list passed in
        # For food in food_positions: check distance, if close -> eat, gain energy, remove food

    def calculate_fitness(self):
        # Example: Simple fitness based on survival time (implicitly tracked) + final energy
        self.fitness = self.energy # Higher energy at the end is better
        # More complex: add points for food eaten, distance covered, etc.


# --- Genetic Algorithm Functions ---
def selection(population):
    # Tournament selection (example)
    parents = []
    for _ in range(POPULATION_SIZE): # Select enough parents for next gen
        tournament = random.sample(population, k=3) # Pick 3 random agents
        winner = max(tournament, key=lambda agent: agent.fitness)
        parents.append(winner)
    return parents

def crossover(parent1, parent2):
    # Single-point crossover (example)
    if len(parent1.genome) != len(parent2.genome):
         # Handle potential length mismatch - maybe return one parent's genome
         print("Warning: Genome length mismatch in crossover")
         return parent1.genome.copy() # Or handle differently

    if len(parent1.genome) < 2:
        return parent1.genome.copy() # Cannot crossover length 1 or 0

    point = random.randint(1, len(parent1.genome) - 1)
    child_genome = np.concatenate((parent1.genome[:point], parent2.genome[point:]))
    return child_genome

def mutate(genome):
    mutated_genome = genome.copy()
    for i in range(len(mutated_genome)):
        if random.random() < MUTATION_RATE:
            mutation = np.random.normal(0, MUTATION_STRENGTH) # Gaussian mutation
            mutated_genome[i] += mutation
            mutated_genome[i] = np.clip(mutated_genome[i], -1, 1) # Keep weights in range
    return mutated_genome

# --- Simulation ---
def run_simulation():
    # 1. Initialization
    population = [Agent() for _ in range(POPULATION_SIZE)]
    # Initialize food positions (needs implementation)
    food_positions = [(random.uniform(0, ENV_WIDTH), random.uniform(0, ENV_HEIGHT)) for _ in range(NUM_FOOD)]

    for generation in range(100): # Number of generations
        print(f"\n--- Generation {generation} ---")

        # Reset agents for the new simulation run within the generation
        for agent in population:
            agent.x = random.uniform(0, ENV_WIDTH)
            agent.y = random.uniform(0, ENV_HEIGHT)
            agent.energy = 100.0
            agent.alive = True
            agent.fitness = 0.0
            # Potentially reset food here as well for each run

        # 2. Simulation Loop (for this generation)
        for step in range(SIMULATION_STEPS):
            active_agents = [agent for agent in population if agent.alive]
            if not active_agents:
                # print(f"All agents died at step {step}")
                break # End early if all agents are dead

            for agent in active_agents:
                agent.update(food_positions)
                # Add logic for eating food here: check distance, update energy, remove food item

            # Optional: Update environment (e.g., respawn food)

        # 3. Evaluation
        total_fitness = 0
        max_fitness = -float('inf')
        for agent in population:
            agent.calculate_fitness() # Calculate based on final state
            total_fitness += agent.fitness
            if agent.fitness > max_fitness:
                max_fitness = agent.fitness
        avg_fitness = total_fitness / POPULATION_SIZE
        print(f"Average Fitness: {avg_fitness:.2f}, Max Fitness: {max_fitness:.2f}")

        # 4. Evolution (Create next generation)
        parents = selection(population)
        next_population = []
        for i in range(0, POPULATION_SIZE, 2): # Assume even population size for pairs
             if i+1 < len(parents):
                 parent1 = parents[i]
                 parent2 = parents[i+1] # Simplistic pairing
                 child1_genome = crossover(parent1, parent2)
                 child2_genome = crossover(parent2, parent1) # Simple swap crossover

                 child1_genome = mutate(child1_genome)
                 child2_genome = mutate(child2_genome)

                 next_population.append(Agent(genome=child1_genome))
                 if len(next_population) < POPULATION_SIZE:
                      next_population.append(Agent(genome=child2_genome))
             elif i < len(parents): # Handle odd population size if needed
                  # Just mutate the last selected parent? Or duplicate?
                  parent = parents[i]
                  child_genome = mutate(parent.genome.copy())
                  next_population.append(Agent(genome=child_genome))


        population = next_population[:POPULATION_SIZE] # Ensure correct population size

# --- Run ---
if __name__ == "__main__":
    # Adjust GENOME_LENGTH based on NN design *before* running
    # Example: 5 inputs, 2 outputs, no hidden layer => weights = 5 * 2 = 10
    # Example: 5 inputs, 4 hidden neurons, 2 outputs => weights = (5*4 + 4) + (4*2 + 2) = 24 + 10 = 34 (including biases)
    # Set a plausible initial GENOME_LENGTH
    num_inputs_example = 5
    num_outputs_example = 2
    # Simple direct connection (no hidden layer, no biases for this basic skeleton)
    GENOME_LENGTH = num_inputs_example * num_outputs_example
    print(f"Initial GENOME_LENGTH set to: {GENOME_LENGTH}")

    run_simulation()
```

**Important Notes & Next Steps:**

1.  **NN Implementation:** The `SimpleNN` class is *highly* simplified. A real implementation needs proper layer definition, matrix multiplication (use `NumPy`), activation functions, and handling of biases. The `GENOME_LENGTH` must precisely match the total number of weights and biases in your chosen network structure. This is critical.
2.  **Sensing:** The `sense` method is just a placeholder. You need to implement logic to calculate actual sensor values (distance/angle to food, walls, etc.).
3.  **Eating:** Logic for agents consuming food is missing and needs to be added in the `update` loop or a dedicated environment interaction step.
4.  **Fitness Function:** The current fitness (`self.energy`) is very basic. Experiment with rewarding food eaten, distance traveled, survival time, etc.
5.  **Genetic Operators:** The `crossover` and `mutate` functions are examples. There are many variations (uniform crossover, different mutation distributions).
6.  **Visualization:** To *see* the evolution, you'll need a graphics library. `Pygame` is good for real-time 2D graphics, or `Matplotlib` can be used to plot agent positions or fitness graphs periodically. This adds significant complexity.
7.  **Debugging:** This kind of simulation is prone to bugs. Print states, check values, start simple and build incrementally. Does the NN output make sense? Do agents move? Does fitness increase *at all* over generations?
8.  **Parameter Tuning:** `MUTATION_RATE`, `MUTATION_STRENGTH`, `POPULATION_SIZE`, `SIMULATION_STEPS`, selection method, NN architecture – all these need tuning for good results.

**Getting Started (Realistically):**

1.  **Focus:** Implement a *very* basic Agent (position only), Environment (just boundaries), and Simulation Loop first. Make the agent move randomly.
2.  **Add Energy:** Give the agent energy, make movement cost energy, make it die at zero.
3.  **Add Food:** Place food, let the agent eat it by proximity, gain energy.
4.  **Implement Basic NN:** Create a *simple* NN structure (e.g., 2 sensors -> 2 outputs). Connect it to agent actions.
5.  **Implement GA:** Add the population, fitness calculation (e.g., survival time), selection, crossover, mutation.
6.  **Iterate & Visualize:** Gradually add complexity (more sensors, better NN, obstacles) and add visualization once the core mechanics work.

This structured approach, starting with the absolute basics and using the provided skeleton, is your best bet. Remember, the 1-hour goal is best viewed as "understand the concepts and have a basic code structure," not a finished product. Good luck!

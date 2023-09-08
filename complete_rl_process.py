
# Starting from scratch: Complete process from the beginning to the end

# Environment Definition
class CompleteFeedbackEnvironment:
    def __init__(self, size=5):
        self.size = size
        self.agent_position = (0, 0)
        self.goal_position = (self.size - 1, self.size - 1)
        self.agent_positions = [self.agent_position]

    def display(self):
        for i in range(self.size):
            row = []
            for j in range(self.size):
                if (i, j) == self.agent_position:
                    row.append("A")
                elif (i, j) == self.goal_position:
                    row.append("G")
                else:
                    row.append("-")
            print(" ".join(row))
        print("\n")

    def move_agent(self, action):
        x, y = self.agent_position
        if action == "N" and x > 0:
            x -= 1
        elif action == "S" and x < self.size - 1:
            x += 1
        elif action == "E" and y < self.size - 1:
            y += 1
        elif action == "W" and y > 0:
            y -= 1

        self.agent_position = (x, y)
        self.agent_positions.append(self.agent_position)

        # Reward structure
        if self.agent_position == self.goal_position:
            return 10
        else:
            return -0.1

    def reset_environment(self):
        self.agent_position = (0, 0)
        self.agent_positions = [self.agent_position]


# Optimistic Greedy Agent Definition
class CompleteOptimisticGreedyAgent:
    def __init__(self, environment, learning_rate=0.1, discount_factor=0.9, optimistic_value=10):
        self.env = environment
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.q_values = {}
        for x in range(self.env.size):
            for y in range(self.env.size):
                for action in ['N', 'S', 'E', 'W']:
                    self.q_values[((x, y), action)] = optimistic_value

    def choose_action(self, state):
        q_values_of_state = {action: self.q_values[(state, action)] for action in ['N', 'S', 'E', 'W']}
        return max(q_values_of_state, key=q_values_of_state.get)

    def learn(self, old_state, action_taken, reward_received, new_state):
        old_q_value = self.q_values[(old_state, action_taken)]
        future_max_q_value = max([self.q_values[(new_state, action)] for action in ['N', 'S', 'E', 'W']])
        self.q_values[(old_state, action_taken)] = old_q_value + self.learning_rate * (reward_received + self.discount_factor * future_max_q_value - old_q_value)

    def take_step(self):
        old_state = self.env.agent_position
        action = self.choose_action(old_state)
        reward = self.env.move_agent(action)
        new_state = self.env.agent_position
        self.learn(old_state, action, reward, new_state)


# Initialization
complete_feedback_env = CompleteFeedbackEnvironment()
complete_greedy_agent = CompleteOptimisticGreedyAgent(complete_feedback_env)

# Phase 1: Exploration and First Goal Achievement (Allow up to 1000 steps for the agent to reach the goal)
for _ in range(1000):
    if complete_feedback_env.agent_position != complete_feedback_env.goal_position:
        complete_greedy_agent.take_step()
    else:
        break

# Phase 2: Refinement and Optimization (Repeated episodes to refine path to goal)
for episode in range(5):  # 5 episodes for demonstration purposes
    complete_feedback_env.reset_environment()
    for _ in range(1000):
        if complete_feedback_env.agent_position != complete_feedback_env.goal_position:
            complete_greedy_agent.take_step()
        else:
            break

# Visualization
positions = complete_feedback_env.agent_positions
x_positions = [pos[0] for pos in positions]
y_positions = [pos[1] for pos in positions]

plt.figure(figsize=(8, 8))
plt.plot(x_positions, y_positions, marker='o', linestyle='-')
plt.scatter(0, 0, color="blue", s=100, label="Start (Agent's initial position)")
plt.scatter(complete_feedback_env.size - 1, complete_feedback_env.size - 1, color="red", s=100, label="Goal")
plt.title("Path Taken by the Agent after Refinement")
plt.xlabel("X Position")
plt.ylabel("Y Position")
plt.legend()
plt.gca().invert_yaxis()
plt.grid(True)
plt.show()

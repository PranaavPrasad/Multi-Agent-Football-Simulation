import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import gymnasium as gym
from collections import deque
import time
import matplotlib.pyplot as plt

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class FootballEnv(gym.Env):
    def __init__(self, grid_rows=121, grid_cols=51):
        super(FootballEnv, self).__init__()
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols

        # Expanded to 9 actions (8 movement directions + block)
        self.action_space = gym.spaces.Discrete(9)
        self.observation_space = gym.spaces.Box(
            low=0, high=1, shape=(7,), dtype=np.float32)

        # Initialize field layout
        self.layout = np.zeros((grid_rows, grid_cols), dtype=str)
        self.layout[:, :] = "."
        self.layout[self.grid_rows//2, self.grid_cols//2] = "C"
        self.layout[self.grid_rows//2-8: self.grid_rows//2+9, -6:-1] = "D"
        self.layout[self.grid_rows//2-8: self.grid_rows//2+9, 0:5] = "d"
        self.layout[:, self.grid_cols//2] = "M"
        self.layout[:, -1] = "O"
        self.layout[:, 0] = "O"
        self.layout[0, :] = "O"
        self.layout[-1, :] = "O"
        self.layout[self.grid_rows//2-4: self.grid_rows//2+5, -1] = "G"
        self.layout[self.grid_rows//2-4: self.grid_rows//2+5, 0] = "g"
        self.ball_pos = (self.grid_rows//2, self.grid_cols//2)
        self.episode_steps = 0

        # Trajectory tracking for visualization
        self.trajectory = []

        # Ball movement parameters (for more realistic ball movement)
        self.ball_speed = 1  # How many cells the ball moves each step
        self.ball_direction = (0, -1)  # Default direction (left)

    def _get_state(self, player, players):
        # Normalized state representation
        same_team = []
        other_team = []
        for play in players:
            if player.team == play.team:
                same_team.extend(
                    [play.position[0]/self.grid_rows, play.position[1]/self.grid_cols])
            else:
                other_team.extend(
                    [play.position[0]/self.grid_rows, play.position[1]/self.grid_cols])

        # Normalize ball position
        ball_row_norm = self.ball_pos[0] / self.grid_rows
        ball_col_norm = self.ball_pos[1] / self.grid_cols

        # Calculate vector from player to ball (direction)
        player_to_ball_row = (
            self.ball_pos[0] - player.position[0]) / self.grid_rows
        player_to_ball_col = (
            self.ball_pos[1] - player.position[1]) / self.grid_cols

        # Return enhanced state with more information
        return np.array([
            player.position[0]/self.grid_rows,  # Player row (normalized)
            player.position[1]/self.grid_cols,  # Player col (normalized)
            ball_row_norm,                      # Ball row (normalized)
            ball_col_norm,                      # Ball col (normalized)
            player_to_ball_row,                 # Direction to ball (row)
            player_to_ball_col,                 # Direction to ball (col)
            float(player.has_ball),             # Has ball flag
        ], dtype=np.float32)

    def reset(self, players, ball_spawn_radius=None, seed=None, options=None):
        super().reset(seed=seed)

        # Clear trajectory
        self.trajectory = []

        # Position players
        occupied_positions = set()
        for player in players:
            while True:
                # Expanded D-box for goalkeeper to have more room to move
                if player.team == 0:  # Left-side goalkeeper
                    # Random starting position within the D-box
                    player_row = random.randint(
                        self.grid_rows // 2 - 10, self.grid_rows // 2 + 10)
                    player_col = random.randint(0, 8)  # Expanded D-box width

                    # Make sure the goalkeeper starts in a good defensive position
                    if random.random() < 0.7:  # 70% of the time, start in central position
                        player_row = self.grid_rows // 2
                        player_col = 1
                elif player.role == "GK":
                    if player.team == 0:
                        player_row = self.grid_rows // 2
                        player_col = 1
                    else:
                        player_row = self.grid_rows // 2
                        player_col = self.grid_cols - 2
                else:
                    player_row = random.randint(
                        self.grid_rows // 2 - 3, self.grid_rows // 2 + 3)
                    player_col = random.randint(
                        3 * self.grid_cols // 4, self.grid_cols - 2)

                if (player_row, player_col) not in occupied_positions:
                    occupied_positions.add((player_row, player_col))
                    player.position = (player_row, player_col)
                    player.has_ball = False
                    break

        # Random ball position within the field
        ball_row = random.randint(1, self.grid_rows - 2)
        ball_col = random.randint(1, self.grid_cols - 2)

        # Keep some distance from left goal
        ball_col = max(10, min(ball_col, self.grid_cols - 2))

        # Make sure ball isn't placed on top of a player
        occupied_positions = [player.position for player in players]
        while (ball_row, ball_col) in occupied_positions:
            ball_row = (ball_row + 1) % (self.grid_rows - 2) + 1

        self.ball_pos = (ball_row, ball_col)

        # Random ball direction with bias towards goal
        if random.random() < 0.8:  # 80% chance of heading toward the goal
            # Calculate direction vector toward the goal (more realistic angles)
            goal_center = (self.grid_rows // 2, 0)
            direction_row = goal_center[0] - ball_row
            direction_col = goal_center[1] - ball_col

            # Normalize and add some randomness
            magnitude = max(1, np.sqrt(direction_row**2 + direction_col**2))
            direction_row = direction_row / \
                magnitude + random.uniform(-0.2, 0.2)
            direction_col = direction_col / \
                magnitude + random.uniform(-0.1, 0.1)

            # Renormalize
            magnitude = max(1, np.sqrt(direction_row**2 + direction_col**2))
            self.ball_direction = (
                direction_row / magnitude, direction_col / magnitude)
        else:
            # Random direction
            # Limit angle to reasonable range
            angle = random.uniform(-np.pi/4, np.pi/4)
            # Heading generally leftward
            self.ball_direction = (np.sin(angle), -np.cos(angle))

        # Randomize ball speed
        self.ball_speed = random.uniform(0.7, 1.3)

        self.episode_steps = 0

        # Add initial positions to trajectory
        for player in players:
            self.trajectory.append({
                'player_pos': player.position,
                'ball_pos': self.ball_pos,
                'has_ball': player.has_ball
            })

        return self._get_state(players[0], players), {}

    def step_goalkeeper(self, action, player, players):
        self.episode_steps += 1
        reward = player.step_penalty  # Small negative reward for each step
        done = False

        # Store previous positions for trajectory
        prev_player_pos = player.position
        prev_ball_pos = self.ball_pos
        prev_has_ball = player.has_ball

        # Move the ball according to its direction and speed
        ball_dx = int(round(self.ball_direction[1] * self.ball_speed))
        ball_dy = int(round(self.ball_direction[0] * self.ball_speed))

        # Ensure ball moves at least one cell per step for better learning
        if ball_dx == 0 and ball_dy == 0:
            ball_dx = -1  # Default left movement if direction would result in no movement

        new_ball_pos = (self.ball_pos[0] + ball_dy, self.ball_pos[1] + ball_dx)

        # Ensure the ball stays in bounds
        if 0 < new_ball_pos[0] < self.grid_rows and 0 < new_ball_pos[1] < self.grid_cols:
            self.ball_pos = new_ball_pos

        # Handle goalkeeper movement (action < 8) or block attempt (action == 8)
        if action < 8:  # Movement actions in 8 directions
            # Expanded movement directions for better control
            dx, dy = [(0, -1), (1, -1), (1, 0), (1, 1), (0, 1),
                       (-1, 1), (-1, 0), (-1, -1)][action]
            # Note: swapped dx, dy to match grid
            new_pos = (player.position[0] + dy, player.position[1] + dx)

            # Check if the new position is within the expanded D box
            # Wider D-box to give goalkeeper more freedom
            d_box_min_row, d_box_max_row = self.grid_rows // 2 - 15, self.grid_rows // 2 + 15
            d_box_min_col, d_box_max_col = 0, 10  # Expanded from 0-5 to 0-10

            is_inside_d_box = (d_box_min_row <= new_pos[0] <= d_box_max_row) and (
                d_box_min_col <= new_pos[1] <= d_box_max_col)

            if is_inside_d_box:
                player.prev_position = player.position
                player.position = new_pos

                # If already has the ball, it moves with the goalkeeper
                if player.has_ball:
                    self.ball_pos = new_pos

                # Calculate distance to ball after movement for reward shaping
                dist_to_ball = np.linalg.norm(
                    np.array(player.position) - np.array(self.ball_pos))

                # Reward for moving toward the ball
                prev_dist = np.linalg.norm(
                    np.array(prev_player_pos) - np.array(self.ball_pos))
                if dist_to_ball < prev_dist:
                    reward += player.move_toward_ball_reward

                # Additional reward for being close to the ball
                if dist_to_ball < 5:
                    reward += player.near_ball_bonus * (1.0 - dist_to_ball/5.0)
            else:
                reward += player.out_of_bounds_penalty
                done = True  # End episode if the goalkeeper moves out of bounds

        # Check for block attempt (action == 8)
        elif action == 8:
            # More forgiving catch radius (increased from 2 to 3)
            catch_radius = 3.0
            dist_to_ball = np.linalg.norm(
                np.array(player.position) - np.array(self.ball_pos))

            if dist_to_ball <= catch_radius:
                # Successful block!
                player.has_ball = True

                # Ball is now with the goalkeeper
                self.ball_pos = player.position

                # Substantial reward for blocking the ball
                reward += player.block_reward + player.ball_held

                # End episode on successful block
                done = True
            else:
                # Failed block attempt - punish slightly to discourage spam blocking
                reward += player.failed_block_penalty

                # But add a small reward based on how close the attempt was
                # (to encourage getting closer for future attempts)
                reward += player.near_miss_reward * \
                    (1.0 - min(1.0, dist_to_ball / 10.0))

        # Check if the ball entered the goal (failure case)
        if 0 <= self.ball_pos[0] < self.grid_rows and self.ball_pos[1] <= 0:
            if self.grid_rows // 2 - 4 <= self.ball_pos[0] <= self.grid_rows // 2 + 4:
                # Ball is in the goal
                reward += player.goal_conceded  # Large negative reward
                done = True

        # End episode if ball goes out of bounds
        if (self.ball_pos[0] <= 0 or self.ball_pos[0] >= self.grid_rows - 1 or
            self.ball_pos[1] <= 0 or self.ball_pos[1] >= self.grid_cols - 1):
            if not player.has_ball:  # Only if goalkeeper didn't catch it
                reward += player.out_of_play_penalty
                done = True

        # Record trajectory for visualization
        self.trajectory.append({
            'player_pos': player.position,
            'ball_pos': self.ball_pos,
            'has_ball': player.has_ball
        })

        # Safety check for maximum episode length
        truncated = self.episode_steps >= 50

        return self._get_state(player, players), reward, done, truncated, {}

    def render(self, players):
        grid = np.full((self.grid_rows, self.grid_cols), '-')

        # Draw field elements
        for i in range(self.grid_rows):
            for j in range(self.grid_cols):
                if self.layout[i, j] == 'O':
                    grid[i, j] = '#'
                elif self.layout[i, j] == 'G':
                    grid[i, j] = '|'
                elif self.layout[i, j] == 'M':
                    grid[i, j] = '.'

        # Draw ball
        if 0 <= self.ball_pos[0] < self.grid_rows and 0 <= self.ball_pos[1] < self.grid_cols:
            grid[self.ball_pos[0], self.ball_pos[1]] = 'B'

        # Draw players
        for player in players:
            if 0 <= player.position[0] < self.grid_rows and 0 <= player.position[1] < self.grid_cols:
                if player.team == 0:
                    grid[player.position[0], player.position[1]] = 'P'
                else:
                    grid[player.position[0], player.position[1]] = 'Q'

        # Print the grid
        print('-' * (self.grid_cols + 2))
        for row in grid:
            print('|' + ''.join(row) + '|')
        print('-' * (self.grid_cols + 2))


class Player():
    def __init__(self, role, team, env):
        self.role = role
        self.team = team
        self.position = [random.randint(
            env.grid_rows//2-3, env.grid_rows//2+3), random.randint(1, env.grid_cols//4)]
        self.prev_position = []
        self.prev_ball_statues = False
        self.has_ball = False

        # Adjusted reward structure to better guide learning
        self.step_penalty = -0.001        # Small penalty per step to encourage efficiency
        self.block_reward = 50.0          # Very large reward for successfully blocking
        self.ball_held = 10.0             # Bonus for holding the ball after block
        self.goal_conceded = -100.0       # Large penalty for conceding a goal
        self.ball_lost = -10.0            # Penalty for losing possession
        self.move_toward_ball_reward = 0.05  # Reward for decreasing distance to ball
        self.near_ball_bonus = 0.1        # Bonus for being near the ball
        self.out_of_bounds_penalty = -5.0  # Penalty for going out of bounds
        self.out_of_play_penalty = -1.0   # Penalty if ball goes out of play
        self.failed_block_penalty = -0.01  # Small penalty for failed block attempts
        self.near_miss_reward = 0.02      # Small reward for near misses to guide learning


class DQN(nn.Module):
    def __init__(self, input_dim, action_size):
        super(DQN, self).__init__()
        # Deeper network architecture
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, action_size)

        # Use better initialization
        for layer in [self.fc1, self.fc2, self.fc3, self.fc4, self.fc5]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        return self.fc5(x)


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size

        # Hyperparameters
        self.gamma = 0.99          # Discount factor
        self.epsilon = 1.0         # Exploration rate
        self.epsilon_min = 0.05    # Minimum exploration rate
        self.epsilon_decay = 0.998  # Slower decay rate for more exploration
        self.learning_rate = 0.0002  # Adjusted learning rate

        # Replay memory
        self.memory = deque(maxlen=100000)
        self.batch_size = 64       # Smaller batch size for more frequent updates
        self.target_update_freq = 10  # More frequent target network updates

        # Neural networks
        self.device = device
        self.model = DQN(state_size, action_size).to(self.device)
        self.target_model = DQN(state_size, action_size).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=self.learning_rate)

        # Tracking metrics
        self.rewards_history = []
        self.loss_history = []
        self.episode_count = 0

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, evaluate=False):
        # Random action during training with probability epsilon
        if not evaluate and np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)

        # Forward pass through neural network
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.model(state)
        return torch.argmax(q_values).item()

    def replay(self):
        if len(self.memory) < self.batch_size:
            return 0

        # Sample random batch from memory
        minibatch = random.sample(self.memory, self.batch_size)

        # Prepare batch data
        states = torch.FloatTensor(
            np.array([experience[0] for experience in minibatch])).to(self.device)
        actions = torch.LongTensor([experience[1]
                                   for experience in minibatch]).to(self.device)
        rewards = torch.FloatTensor([experience[2]
                                    for experience in minibatch]).to(self.device)
        next_states = torch.FloatTensor(
            np.array([experience[3] for experience in minibatch])).to(self.device)
        dones = torch.FloatTensor([experience[4]
                                  for experience in minibatch]).to(self.device)

        # Current Q values
        curr_q_values = self.model(states).gather(1, actions.unsqueeze(1))

        # Target Q values
        with torch.no_grad():
            next_q_values = self.target_model(next_states).max(1)[0]

        # Compute target values using Bellman equation
        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        # Compute loss
        loss = F.mse_loss(curr_q_values.squeeze(), target_q_values)

        # Gradient descent step
        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

        self.optimizer.step()

        # Record loss for tracking
        self.loss_history.append(loss.item())

        return loss.item()

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


def curriculum_train_goalkeeper(env, episodes=10000, stages=5, episodes_per_stage=100, max_steps=50):
    """
    Train goalkeeper agent using curriculum learning with increasing difficulty.

    Args:
        env: Football environment
        episodes: Total number of episodes for training
        stages: Number of curriculum difficulty stages
        episodes_per_stage: Episodes per curriculum stage
        max_steps: Maximum steps per episode

    Returns:
        Trained DQN agent
    """
    # Create a goalkeeper
    players = [Player('GK', 0, env)]

    # Get initial state to determine state size
    env.reset(players)
    sample_state = env._get_state(players[0], players)
    state_size = len(sample_state)

    print(f"State size: {state_size}")

    # Create a DQN agent with the determined state size
    agent = DQNAgent(state_size, 9)  # 9 possible actions (8 movements + block)

    # Define curriculum parameters
    # Start with slower, more predictable balls at closer distances
    # Gradually increase speed, distance, and randomness
    for stage in range(stages):
        stage_difficulty = (stage + 1) / stages  # 0.2, 0.4, 0.6, 0.8, 1.0

        print(
            f"\n--- Starting Curriculum Stage {stage+1}/{stages} (Difficulty: {stage_difficulty:.2f}) ---")

        # Ball distance increases with difficulty
        min_distance = 5 + stage_difficulty * 15  # 5-20 range
        max_distance = min_distance + 5

        # Ball speed increases with difficulty
        ball_speed = 0.5 + stage_difficulty * 0.5  # 0.5-1.0 range

        # Randomness increases with difficulty
        angle_variance = stage_difficulty * 0.6  # 0-0.6 range

        for episode in range(episodes_per_stage):
            # Reset the environment
            env.reset(players)

            # Place ball at specified distance from goal with appropriate angle
            goal_center_row = env.grid_rows // 2
            ball_distance = random.uniform(min_distance, max_distance)
            angle = random.uniform(-angle_variance, angle_variance)

            # Calculate ball position
            ball_row = goal_center_row + int(ball_distance * np.sin(angle))
            ball_col = 1 + ball_distance  # Distance from left goal line

            # Ensure ball stays in bounds
            ball_row = max(1, min(ball_row, env.grid_rows - 2))
            ball_col = max(10, min(ball_col, env.grid_cols - 2))

            # Set ball position
            env.ball_pos = (ball_row, ball_col)

            # Set ball direction toward goal with appropriate angle
            goal_center = (env.grid_rows // 2, 0)
            direction_row = goal_center[0] - ball_row
            direction_col = goal_center[1] - ball_col

            # Add randomness based on stage
            direction_row += random.uniform(-angle_variance, angle_variance)
            direction_col += random.uniform(-angle_variance *
                                            0.5, angle_variance * 0.5)

            # Normalize direction vector
            magnitude = max(0.1, np.sqrt(direction_row**2 + direction_col**2))
            env.ball_direction = (direction_row / magnitude,
                                  direction_col / magnitude)

            # Set ball speed
            env.ball_speed = ball_speed * random.uniform(0.8, 1.2)

            total_reward = 0
            done = False
            truncated = False

            # Get initial state
            state = env._get_state(players[0], players)

            for step in range(max_steps):
                if done or truncated:
                    break

                # Choose action
                action = agent.act(state)

                # Take action
                next_state, reward, done, truncated, _ = env.step_goalkeeper(
                    action, players[0], players)

                # Remember experience
                agent.remember(state, action, reward,
                               next_state, done or truncated)

                # Learn from batches of experience
                if len(agent.memory) >= agent.batch_size:
                    loss = agent.replay()

                state = next_state
                total_reward += reward

            # Track rewards
            agent.rewards_history.append(total_reward)

            # Update target network periodically
            agent.episode_count += 1
            if agent.episode_count % agent.target_update_freq == 0:
                agent.update_target_model()

            # Decay exploration rate
            agent.decay_epsilon()

            # Show progress
            if (episode + 1) % 10 == 0:
                avg_reward = np.mean(agent.rewards_history[-10:])
                print(
                    f"Stage {stage+1}, Episode {episode+1}: Reward = {total_reward:.2f}, Avg(10) = {avg_reward:.2f}, Epsilon = {agent.epsilon:.3f}")

        # Evaluate at the end of each curriculum stage
        eval_rewards = []
        for _ in range(5):  # 5 evaluation episodes
            env.reset(players)
            eval_total_reward = 0
            eval_done = False
            eval_truncated = False
            eval_state = env._get_state(players[0], players)

            for step in range(max_steps):
                if eval_done or eval_truncated:
                    break

                # Use greedy policy for evaluation
                eval_action = agent.act(eval_state, evaluate=True)
                eval_next_state, eval_reward, eval_done, eval_truncated, _ = env.step_goalkeeper(
                    eval_action, players[0], players)
                eval_state = eval_next_state
                eval_total_reward += eval_reward

            eval_rewards.append(eval_total_reward)

        print(
            f"Stage {stage+1} Evaluation: Avg Reward = {np.mean(eval_rewards):.2f}")

    print("\nTraining complete!")
    return agent


def evaluate_goalkeeper_agent(agent, env, episodes=5, max_steps=50, render=True):
    """
    Evaluate goalkeeper agent performance.

    Args:
        agent: Trained DQN agent
        env: Football environment
        episodes: Number of evaluation episodes
        max_steps: Maximum steps per episode
        render: Whether to render the environment
    """
    # Create a goalkeeper
    players = [Player('GK', 0, env)]

    episode_rewards = []
    success_count = 0

    for episode in range(episodes):
        # Reset the environment
        env.reset(players)

        # Place ball at realistic position
        ball_row = random.randint(
            env.grid_rows // 2 - 10, env.grid_rows // 2 + 10)
        ball_col = random.randint(10, 30)  # Medium distance
        env.ball_pos = (ball_row, ball_col)

        # Direction toward goal
        goal_center = (env.grid_rows // 2, 0)
        direction_row = goal_center[0] - ball_row
        direction_col = goal_center[1] - ball_col

        # Normalize direction
        magnitude = np.sqrt(direction_row**2 + direction_col**2)
        env.ball_direction = (direction_row / magnitude,
                              direction_col / magnitude)

        # Random but reasonable speed
        env.ball_speed = random.uniform(0.8, 1.2)

        total_reward = 0
        done = False
        truncated = False

        # Get initial state
        state = env._get_state(players[0], players)

        for step in range(max_steps):
            if done or truncated:
                break

            # Render if requested
            if render:
                env.render(players)
                time.sleep(0.1)  # Slow down for visualization

            # Choose action
            action = agent.act(state, evaluate=True)
            print(f"Step {step}, Action: {action}")

            # Take action
            next_state, reward, done, truncated, _ = env.step_goalkeeper(
                action, players[0], players)

            state = next_state
            total_reward += reward

        # Final render to show result
        if render:
            env.render(players)

        # Track results
        episode_rewards.append(total_reward)

        # Count successes (goalkeeper caught the ball)
        if players[0].has_ball:
            success_count += 1

        print(
            f"Episode {episode+1}: Reward = {total_reward:.2f}, Success = {players[0].has_ball}")

    # Print final results
    print(f"\nEvaluation complete!")
    print(f"Average reward: {np.mean(episode_rewards):.2f}")
    print(f"Success rate: {success_count / episodes:.2f}")

    return episode_rewards

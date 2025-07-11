# Tic-Tac-Toe with Q-Learning

This project implements a Tic-Tac-Toe game where an AI agent learns to play using reinforcement learning, specifically the Q-Learning algorithm. The program supports both terminal and graphical interfaces, allowing human players to compete against the AI.

## Q-Learning Theory

Q-Learning is a reinforcement learning algorithm that enables an agent to learn optimal actions in a given environment by interacting with it. The agent uses a Q-table to store the expected rewards for each state-action pair. The algorithm updates the Q-values iteratively based on the following formula:

```
Q(state, action) = Q(state, action) + α * (reward + γ * max(Q(next_state, all_actions)) - Q(state, action))
```

Where:
- `α` (alpha) is the learning rate, controlling how much new information overrides old information.
- `γ` (gamma) is the discount factor, determining the importance of future rewards.
- `reward` is the immediate reward received after taking an action.
- `max(Q(next_state, all_actions))` is the maximum expected reward for the next state.

Through repeated interactions, the agent learns to maximize its cumulative reward by choosing actions that lead to better outcomes. In this project, Q-Learning is applied to train an AI agent to play Tic-Tac-Toe optimally.


## Features

- **Random Policy**: An agent that chooses moves randomly.
- **Q-Learning Policy**: An agent that learns optimal moves through reinforcement learning.
- **Terminal Interface**: Play the game in the terminal.
- **Graphical Interface**: Play the game with a GUI using PyQt5.

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd q-learning-tic-tac-toe
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

   If you want to use the graphical interface, ensure PyQt5 is installed:
   ```bash
   pip install PyQt5
   ```

## Usage

### Terminal Interface

To play against the Random Policy:
```bash
python main.py
```

To play against the Q-Learning Policy (after training):
```bash
python main.py -p QLearningPolicy
```

### Graphical Interface

To play against the Random Policy:
```bash
python main_gui.py
```
Select "Random" as your opponent in the GUI.

To play against the Q-Learning Policy (after training):
```bash
python main_gui.py
```
Select "Q-Learning" as your opponent in the GUI.

## Training the Q-Learning Agent

To train the Q-Learning agent, run:
```bash
python train.py -n <number_of_games>
```
Replace `<number_of_games>` with the number of games you want the agent to train on.

During training, the program will periodically test the agent's performance against a Random Policy and display win, lose, and tie rates.

## File Structure

- `tic_tac_toe.py`: Implements the Tic-Tac-Toe board and game logic.
- `policy.py`: Contains the Random Policy and Q-Learning Policy implementations.
- `train.py`: Script for training the Q-Learning agent.
- `main.py`: Terminal interface for playing the game.
- `main_gui.py`: Graphical interface for playing the game.
- `requirements.txt`: Lists required Python libraries.
- `assets_gui/`: Contains GUI assets like `.ui` files and themes.

## Notes

- The Q-Learning agent saves its learned Q-table to `q_table_player1` and `q_table_player2` files. These files are loaded automatically when the agent is initialized.
- The graphical interface requires PyQt5. Ensure it is installed before running `main_gui.py`.



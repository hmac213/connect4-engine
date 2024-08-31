import numpy as np
from connect4 import Connect4
from model import Connect4NN
from mcts import MCTSNode, MCTS
import os
import tensorflow as tf

# TODO: Create random state generation for varied starting states: pick a piece count randomly (0 - 42?) and make moves until the piece count is reached. If the game is over, undo the move and make another until a move cannot be made.

def self_play(mcts, tau, num_games=100):
    red_wins = 0
    yellow_wins = 0
    draws = 0
    training_data = []

    for game in range(num_games):
        print(f"playing game {game + 1} of this iteration")
        game_history = []
        state = Connect4()
        start_from_random = np.random.rand() < 0.2

        if start_from_random:
            state.random_start()
            print("doing a random start")

        while not state.game_over():
            policy = mcts.search(state, temp=tau)

            if tau == 0:
                max_prob = np.max(policy)
                max_indices = np.where(policy == max_prob)[0]
                if len(max_indices) > 1:
                    col = np.random.choice(max_indices)
                else:
                    col = max_indices[0]
            else:
                col = np.random.choice(len(policy), p=policy)

            player_turn = state.turn  # Track the current player
            state = state.place_piece(col, player_turn)
            state.print_board()
            
            # Record the board state, policy, and the player who made the move
            game_history.append((state.get_board(), policy, player_turn))

        outcome = state.get_result()

        if outcome == 1:
            red_wins += 1
        elif outcome == 2:
            yellow_wins += 1
        else:
            draws += 1

        for board_state, policy, player in game_history:
            if outcome == 1:  # Red wins
                value = 1 if player == 1 else -1
            elif outcome == 2:  # Yellow wins
                value = -1 if player == 1 else 1
            else:  # Draw
                value = 0
            
            flipped_board_state = np.flip(board_state, axis=1)
            flipped_policy = np.flip(policy)

            training_data.append((board_state, policy, value))
            training_data.append((flipped_board_state, flipped_policy, value))

    print(f"Red wins: {red_wins}, Yellow wins: {yellow_wins}, Draws: {draws}")
    return training_data

def prepare_training_data(training_data):
    states = []
    policies = []
    values = []

    for board_state, policy, outcome in training_data:
        states.append(board_state)
        policies.append(policy)
        values.append(outcome)

    states = np.array(states).reshape(-1, 6, 7, 1)  # Reshape for the neural network
    policies = np.array(policies)
    values = np.array(values).reshape(-1, 1)  # Reshape to match the output layer

    return states, policies, values

def train_neural_network(model, states, policies, values, batch_size=64, epochs=10):
    model.fit(
        x=states,
        y={'policy_output': policies, 'value_output': values},
        batch_size=batch_size,
        epochs=epochs
    )

initial_tau = 2
final_tau = 0.5

for iteration in range(10):
    model_file = f'connect4_model_iteration_{iteration}.keras'
    
    if os.path.exists(model_file):
        new_model = tf.keras.models.load_model(model_file)
        print("using an existing file")
    else:
        new_model = Connect4NN(6, 7).model
        print("no existing model. loading new one")

    mcts = MCTS(new_model, 400)

    print(f"Iteration {iteration + 1}")
    
    tau = initial_tau - (iteration / 10) * (initial_tau - final_tau)

    # Self-play to generate new training data
    training_data = self_play(mcts, tau=tau, num_games=100)
    
    # Prepare the data
    states, policies, values = prepare_training_data(training_data)
    
    # Train the neural network
    train_neural_network(new_model, states, policies, values, batch_size=32, epochs=10)
    
    # Optionally, save the model weights after each iteration
    new_model.save(f'connect4_model_iteration_{iteration + 1}.keras')
    
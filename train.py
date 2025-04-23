import numpy as np
from connect4 import Connect4
from model import Connect4NN
from mcts import MCTSNode, MCTS
import os
import tensorflow as tf
from datetime import datetime
import json

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

            player_turn = state.turn
            state = state.place_piece(col, player_turn)
            state.print_board()
            
            game_history.append((state.get_board(), policy, player_turn))

        outcome = state.get_result()

        if outcome == 1:
            red_wins += 1
        elif outcome == 2:
            yellow_wins += 1
        else:
            draws += 1

        for board_state, policy, player in game_history:
            if outcome == 1:
                value = 1 if player == 1 else -1
            elif outcome == 2:
                value = -1 if player == 1 else 1
            else:
                value = 0
            
            flipped_board_state = np.flip(board_state, axis=1)
            flipped_policy = np.flip(policy)

            training_data.append((board_state, policy, value))
            training_data.append((flipped_board_state, flipped_policy, value))

    print(f"Red wins: {red_wins}, Yellow wins: {yellow_wins}, Draws: {draws}")
    return training_data, {"red_wins": red_wins, "yellow_wins": yellow_wins, "draws": draws}

def prepare_training_data(training_data):
    states = []
    policies = []
    values = []

    for board_state, policy, player in training_data:
        # Create 3-channel input representation
        red_channel = (board_state == 1).astype(float)
        yellow_channel = (board_state == 2).astype(float)
        current_player_channel = np.ones_like(board_state) * (1 if player == 1 else 0)
        
        # Stack channels
        state = np.stack([red_channel, yellow_channel, current_player_channel], axis=-1)
        
        states.append(state)
        policies.append(policy)
        values.append(player)

    states = np.array(states)
    policies = np.array(policies)
    values = np.array(values).reshape(-1, 1)

    return states, policies, values

def evaluate_model(model, mcts, num_games=50):
    """Evaluate the model by playing games against itself"""
    results = []
    for _ in range(num_games):
        state = Connect4()
        while not state.game_over():
            # Prepare input for model
            board_state = state.get_board()
            red_channel = (board_state == 1).astype(float)
            yellow_channel = (board_state == 2).astype(float)
            current_player_channel = np.ones_like(board_state) * (1 if state.turn == 1 else 0)
            model_input = np.stack([red_channel, yellow_channel, current_player_channel], axis=-1)
            model_input = np.expand_dims(model_input, axis=0)
            
            policy = mcts.search(state, temp=0)  # Use temp=0 for evaluation
            col = np.argmax(policy)
            state = state.place_piece(col, state.turn)
        results.append(state.get_result())
    
    red_wins = results.count(1)
    yellow_wins = results.count(2)
    draws = results.count(3)
    
    return {
        "red_wins": red_wins,
        "yellow_wins": yellow_wins,
        "draws": draws,
        "win_rate": (red_wins + yellow_wins) / num_games
    }

def train_neural_network(model, states, policies, values, batch_size=64, epochs=10, validation_split=0.1):
    # Create TensorBoard callback
    log_dir = f"logs/fit/{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir,
        histogram_freq=1,
        write_graph=True,
        write_images=True
    )

    # Create early stopping callback
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True
    )

    # Create model checkpoint callback
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath='best_model.keras',
        monitor='val_loss',
        save_best_only=True,
        save_weights_only=False
    )

    history = model.fit(
        x=states,
        y={'policy_output': policies, 'value_output': values},
        batch_size=batch_size,
        epochs=epochs,
        validation_split=validation_split,
        callbacks=[tensorboard_callback, early_stopping, checkpoint_callback]
    )

    return history

def main():
    initial_tau = 2
    final_tau = 0.5
    num_iterations = 10
    training_log = []

    for iteration in range(num_iterations):
        print(f"\nStarting iteration {iteration + 1}/{num_iterations}")
        
        # Load or create model
        model_file = f'connect4_model_iteration_{iteration}.keras'
        if os.path.exists(model_file):
            model = tf.keras.models.load_model(model_file)
            print("Loaded existing model")
        else:
            model = Connect4NN(6, 7).model
            print("Created new model")

        mcts = MCTS(model, 400)
        tau = initial_tau - (iteration / num_iterations) * (initial_tau - final_tau)

        # Generate training data
        print("Generating training data...")
        training_data, game_stats = self_play(mcts, tau=tau, num_games=100)
        states, policies, values = prepare_training_data(training_data)

        # Train the model
        print("Training model...")
        history = train_neural_network(model, states, policies, values, batch_size=32, epochs=10)

        # Evaluate the model
        print("Evaluating model...")
        eval_results = evaluate_model(model, mcts)

        # Log iteration results
        iteration_log = {
            "iteration": iteration + 1,
            "training_games": game_stats,
            "evaluation": eval_results,
            "training_metrics": {
                "policy_loss": history.history['policy_output_loss'][-1],
                "value_loss": history.history['value_output_loss'][-1],
                "policy_accuracy": history.history['policy_output_accuracy'][-1],
                "value_mse": history.history['value_output_mse'][-1]
            }
        }
        training_log.append(iteration_log)

        # Save the model
        model.save(f'connect4_model_iteration_{iteration + 1}.keras')
        
        # Save training log
        with open('training_log.json', 'w') as f:
            json.dump(training_log, f, indent=2)

        print(f"\nIteration {iteration + 1} complete")
        print(f"Training metrics: {iteration_log['training_metrics']}")
        print(f"Evaluation results: {iteration_log['evaluation']}")

if __name__ == "__main__":
    main()
    
import numpy as np
from connect4 import Connect4
from model import Connect4NN
import os
import tensorflow as tf
from datetime import datetime
import json
import multiprocessing as mp
from functools import partial

# Enable unsafe deserialization for Lambda layers
tf.keras.config.enable_unsafe_deserialization()

# TODO: Create random state generation for varied starting states: pick a piece count randomly (0 - 42?) and make moves until the piece count is reached. If the game is over, undo the move and make another until a move cannot be made.

def play_game(model, tau=1.0, batch_size=32, initial_state=None):
    """Play a single game using direct policy predictions with batched inference"""
    game_history = []
    state = initial_state if initial_state is not None else Connect4()
    
    # Increased random start probability (50% chance)
    if initial_state is None and np.random.random() < 0.5:
        state.random_start()

    # Pre-calculate temperature decay factors
    move_count = 0
    # Adjusted temperature to balance exploration and exploitation
    min_temp = 0.5  # Reduced from 0.8 for sharper policy
    
    # Simplified value tracking (only last 3 positions)
    last_values = [0, 0, 0]  # Initialize with neutral values
    value_idx = 0
    
    while not state.game_over():
        # Slower temperature decay to encourage exploration early, but sharper decisions later
        current_tau = tau * max(min_temp, 1.0 - (move_count / 30))  # Faster decay (40 → 30)
        
        # Prepare input (no change - this is already efficient)
        board = state.get_board()
        red_channel = (board == 1).astype(np.float32)
        yellow_channel = (board == 2).astype(np.float32)
        current_player_channel = np.ones_like(board, dtype=np.float32) * (1 if state.turn == 1 else 0)
        state_input = np.stack([red_channel, yellow_channel, current_player_channel], axis=-1)
        state_input = np.expand_dims(state_input, axis=0)

        # Get policy and value prediction
        policy, value = model.predict(state_input, verbose=0)
        policy = policy[0]
        value = value[0][0]
        
        # Efficient value tracking with circular buffer
        last_values[value_idx] = value
        value_idx = (value_idx + 1) % 3
        avg_value = sum(last_values) / 3

        # Get valid moves
        valid_moves = [i for i in range(7) if state.is_valid_move(i)]
        
        # Random move with small probability to ensure exploration, but only very early in the game
        if np.random.random() < 0.03 and move_count < 5:  # Reduced from 0.05 and 10 moves to 0.03 and 5 moves
            col = np.random.choice(valid_moves)
            player_turn = state.turn
            state = state.place_piece(col, player_turn)
            game_history.append((state.get_board(), policy, player_turn))
            move_count += 1
            continue
        
        # Dirichlet noise with reduced impact for better learning signal
        noise_alpha = 0.3 if move_count < 15 else 0.5  # Reduced from 0.5 and 20 moves
        noise = np.random.dirichlet([noise_alpha] * len(valid_moves))
        
        # Reduced noise weight for sharper policies
        noise_weight = 0.2  # Reduced from 0.35
        
        # Vectorized policy modification
        valid_policy = np.full(7, -np.inf)
        valid_policy[valid_moves] = (1 - noise_weight) * policy[valid_moves] + noise_weight * noise
        
        # Faster policy normalization
        valid_policy = np.exp(valid_policy / current_tau)
        valid_policy /= valid_policy.sum()

        # Choose move - increased stochasticity at beginning, but more deterministic later
        if move_count < 10 or np.random.random() < 0.15:  # Reduced from 15 moves and 0.2 probability
            col = np.random.choice(7, p=valid_policy)
        else:
            col = valid_moves[np.argmax([valid_policy[m] for m in valid_moves])]

        # Make move
        player_turn = state.turn
        state = state.place_piece(col, player_turn)
        game_history.append((state.get_board(), policy, player_turn))
        move_count += 1

    return game_history, state.get_result()

def process_game_result(game_history, outcome):
    """Process a single game's history into training data with proper normalization"""
    training_data = []
    
    # Calculate the effective game length for proper value decay
    game_length = len(game_history)
    
    for idx, (board_state, policy, player) in enumerate(game_history):
        # Calculate position value with temporal difference
        if outcome == 1:  # Red wins
            base_value = 1 if player == 1 else -1
        elif outcome == 2:  # Yellow wins
            base_value = -1 if player == 1 else 1
        else:  # Draw
            base_value = 0
            
        # Apply temporal difference - positions closer to the end have more accurate values
        moves_to_end = game_length - idx
        value = base_value * (0.99 ** moves_to_end)  # Discount factor of 0.99
        
        # Add original position
        red_channel = (board_state == 1).astype(float)
        yellow_channel = (board_state == 2).astype(float)
        current_player_channel = np.ones_like(board_state) * (1 if player == 1 else 0)
        state = np.stack([red_channel, yellow_channel, current_player_channel], axis=-1)
        
        # Convert raw policy to proper probability distribution
        valid_moves = [i for i in range(7) if i in np.where(policy > -np.inf)[0]]
        if valid_moves:
            # Normalize only valid moves
            policy_probs = np.zeros_like(policy)
            policy_probs[valid_moves] = policy[valid_moves]
            policy_probs = np.exp(policy_probs)
            
            # Apply temperature scaling to increase sharpness of winning move distributions
            # Use lower temperature (sharper distribution) for moves near the end of winning games
            if outcome != 3 and moves_to_end < 10:  # Not a draw and near end
                # Determine if this move was made by the winner
                is_winner_move = (outcome == 1 and player == 1) or (outcome == 2 and player == 2)
                
                if is_winner_move:
                    # Sharpen policy for winning player's moves near end of game
                    temperature = max(0.5, 1.0 - (10 - moves_to_end) * 0.05)
                    policy_probs = np.power(policy_probs, 1/temperature)
            
            policy_probs = policy_probs / policy_probs.sum()
        else:
            # Uniform distribution if no valid moves (shouldn't happen in normal play)
            policy_probs = np.ones(7) / 7
            
        training_data.append((state, policy_probs, value))
        
        # Add mirrored position with proper policy mirroring
        flipped_state = np.flip(state, axis=1)
        flipped_policy = np.flip(policy_probs)
        training_data.append((flipped_state, flipped_policy, value))
        
        # Add random noise to data for regularization (to prevent memorization)
        # Reduce the amount of noise for more consistent learning
        if np.random.random() < 0.15:  # Reduced from 20% to 15% chance
            noise_state = state.copy()
            # Add small Gaussian noise to the state channels
            noise_state += np.random.normal(0, 0.03, noise_state.shape)  # Reduced noise magnitude
            # Keep the positions of pieces unchanged - just add noise to values
            noise_state = np.clip(noise_state, 0, 1)
            # Add smaller noise to policy
            noise_policy = policy_probs * (1 + np.random.normal(0, 0.05, policy_probs.shape))  # Reduced from 0.1
            noise_policy = noise_policy / noise_policy.sum()
            # Add smaller noise to value
            noise_value = value + np.random.normal(0, 0.05)  # Reduced from 0.1
            noise_value = np.clip(noise_value, -1, 1)
            training_data.append((noise_state, noise_policy, noise_value))
    
    return training_data

def generate_training_data(model, num_games):
    """Generate training data with balanced positions using batched predictions"""
    # Pre-allocate lists with adjusted size estimate
    estimated_moves_per_game = 35  # Adjusted based on actual data
    estimated_total_positions = num_games * estimated_moves_per_game * 2  # *2 for mirrored positions
    all_training_data = []
    
    game_stats = {"red_wins": 0, "yellow_wins": 0, "draws": 0}
    
    # Process games in batches for efficiency
    batch_size = 32  # Number of games to process simultaneously
    num_batches = (num_games + batch_size - 1) // batch_size
    
    for batch_idx in range(num_batches):
        start_game = batch_idx * batch_size
        end_game = min(start_game + batch_size, num_games)
        current_batch_size = end_game - start_game
        
        # Initialize states for all games in batch
        states = [Connect4() for _ in range(current_batch_size)]
        game_histories = [[] for _ in range(current_batch_size)]
        move_counts = [0] * current_batch_size
        last_values = [[0, 0, 0] for _ in range(current_batch_size)]  # Value tracking for each game
        value_indices = [0] * current_batch_size
        active_games = list(range(current_batch_size))
        
        # Random start for some games (20% chance)
        for i in range(current_batch_size):
            if np.random.random() < 0.2:
                states[i].random_start()
        
        while active_games:
            # Prepare batch input for all active games
            batch_inputs = []
            for game_idx in active_games:
                board = states[game_idx].get_board()
                red_channel = (board == 1).astype(np.float32)
                yellow_channel = (board == 2).astype(np.float32)
                current_player_channel = np.ones_like(board, dtype=np.float32) * (1 if states[game_idx].turn == 1 else 0)
                game_input = np.stack([red_channel, yellow_channel, current_player_channel], axis=-1)
                batch_inputs.append(game_input)
            
            # Single model prediction for all active games
            batch_inputs = np.stack(batch_inputs, axis=0)
            policies, values = model.predict(batch_inputs, verbose=0, batch_size=len(active_games))
            
            # Process each active game
            new_active_games = []
            for i, game_idx in enumerate(active_games):
                state = states[game_idx]
                policy = policies[i]
                value = values[i][0]
                move_count = move_counts[game_idx]
                
                # Update value tracking
                last_values[game_idx][value_indices[game_idx]] = value
                value_indices[game_idx] = (value_indices[game_idx] + 1) % 3
                avg_value = sum(last_values[game_idx]) / 3
                
                # Calculate temperature
                current_tau = 1.0 * max(0.7, 1.0 - (move_count / 35))
                
                # Get valid moves and calculate noise
                valid_moves = [i for i in range(7) if state.is_valid_move(i)]
                noise_alpha = 0.3 if move_count < 20 else 0.5
                noise = np.random.dirichlet([noise_alpha] * len(valid_moves))
                
                # Calculate noise weight based on position strength
                noise_weight = 0.35 if (state.turn == 1 and avg_value > 0.3) or \
                                    (state.turn == 2 and avg_value < -0.3) else 0.25
                
                # Calculate move probabilities
                valid_policy = np.full(7, -np.inf)
                valid_policy[valid_moves] = (1 - noise_weight) * policy[valid_moves] + noise_weight * noise
                valid_policy = np.exp(valid_policy / current_tau)
                valid_policy /= valid_policy.sum()
                
                # Choose and make move
                col = np.random.choice(7, p=valid_policy)
                player_turn = state.turn
                states[game_idx] = state.place_piece(col, player_turn)
                game_histories[game_idx].append((state.get_board(), policy, player_turn))
                move_counts[game_idx] += 1
                
                # Check if game continues
                if not states[game_idx].game_over():
                    new_active_games.append(game_idx)
                else:
                    # Process completed game
                    result = states[game_idx].get_result()
                    if result == 1:
                        game_stats["red_wins"] += 1
                    elif result == 2:
                        game_stats["yellow_wins"] += 1
                    else:
                        game_stats["draws"] += 1
                    
                    # Process game history
                    training_data = process_game_result(game_histories[game_idx], result)
                    all_training_data.extend(training_data)
            
            active_games = new_active_games
        
        # Print progress
        if (batch_idx + 1) % max(1, num_batches // 10) == 0:
            games_completed = min((batch_idx + 1) * batch_size, num_games)
            print(f"Completed {games_completed}/{num_games} games")
            print(f"Current stats: {game_stats}")
            print(f"Positions collected: {len(all_training_data)}/{estimated_total_positions}")
            total_games = sum(game_stats.values())
            if total_games > 0:
                print(f"Red win rate: {game_stats['red_wins']/total_games:.1%}")
                print(f"Yellow win rate: {game_stats['yellow_wins']/total_games:.1%}")
                print(f"Draw rate: {game_stats['draws']/total_games:.1%}")
    
    return all_training_data, game_stats

def prepare_training_data(training_data):
    """Prepare training data for the neural network"""
    states = []
    policies = []
    values = []

    for state, policy, value in training_data:
        states.append(state)
        policies.append(policy)
        values.append(value)

    # Convert to numpy arrays with explicit dtype
    return np.array(states, dtype=np.float32), np.array(policies, dtype=np.float32), np.array(values, dtype=np.float32).reshape(-1, 1)

def evaluate_model(model, num_games=50):
    """Evaluate the model by playing games against itself with batched predictions"""
    results = []
    games_in_progress = []
    max_moves_per_game = 42  # Maximum possible moves in Connect4
    move_counter = {}  # Track moves per game
    
    # Start all games
    for _ in range(num_games):
        state = Connect4()
        if np.random.rand() < 0.5:  # 50% chance of random start for diverse evaluation
            state.random_start()
        games_in_progress.append(state)
        move_counter[id(state)] = 0
    
    # Play all games to completion
    while games_in_progress:
        # Prepare batch of current states
        batch_states = []
        batch_indices = []
        
        for idx, state in enumerate(games_in_progress):
            board = state.get_board()
            red_channel = (board == 1).astype(np.float32)
            yellow_channel = (board == 2).astype(np.float32)
            current_player_channel = np.ones_like(board, dtype=np.float32) * (1 if state.turn == 1 else 0)
            state_input = np.stack([red_channel, yellow_channel, current_player_channel], axis=-1)
            batch_states.append(state_input)
            batch_indices.append(idx)
        
        # Get predictions for all states at once
        if batch_states:
            try:
                batch_input = np.stack(batch_states, axis=0)
                policies, _ = model.predict(batch_input, verbose=0, batch_size=len(batch_states))
                
                # Process each game
                new_games_in_progress = []
                for i, (idx, policy) in enumerate(zip(batch_indices, policies)):
                    state = games_in_progress[idx]
                    move_counter[id(state)] += 1
                    
                    # Check for timeout or stuck game
                    if move_counter[id(state)] > max_moves_per_game:
                        print(f"Game {len(results)} exceeded maximum moves, counting as draw")
                        results.append(3)  # Count as draw
                        continue
                    
                    # During evaluation, introduce some randomness (20% of the time)
                    # This helps evaluate true policy strength rather than deterministic patterns
                    use_random = np.random.random() < 0.2 and move_counter[id(state)] < 10
                    
                    # Make move 
                    valid_moves = [i for i in range(7) if state.is_valid_move(i)]
                    if not valid_moves:
                        results.append(3)  # Draw if no valid moves
                        continue
                        
                    # Filter policy for only valid moves
                    valid_policy = np.array([policy[i] if i in valid_moves else -np.inf for i in range(7)])
                    
                    if use_random:
                        # Apply softmax with temperature
                        valid_policy = np.exp(valid_policy / 0.8)
                        valid_policy = valid_policy / valid_policy.sum()
                        col = np.random.choice(7, p=valid_policy)
                    else:
                        col = np.argmax(valid_policy)
                    
                    # Make move
                    player_turn = state.turn
                    try:
                        new_state = state.place_piece(col, player_turn)
                        
                        if new_state.game_over():
                            results.append(new_state.get_result())
                            if len(results) % 10 == 0:
                                print(f"Evaluated {len(results)}/{num_games} games")
                                print(f"Current results: Red wins: {results.count(1)}, Yellow wins: {results.count(2)}, Draws: {results.count(3)}")
                        else:
                            new_games_in_progress.append(new_state)
                            move_counter[id(new_state)] = move_counter[id(state)]
                    except Exception as e:
                        print(f"Error making move: {e}, counting as draw")
                        results.append(3)
                
                games_in_progress = new_games_in_progress
                
            except Exception as e:
                print(f"Error in batch prediction: {e}")
                # If we get an error, count remaining games as draws
                results.extend([3] * (num_games - len(results)))
                break
    
    # Calculate statistics
    stats = {
        "red_wins": results.count(1),
        "yellow_wins": results.count(2),
        "draws": results.count(3),
        "win_rate": (results.count(1) + results.count(2)) / num_games
    }
    
    print(f"\nEvaluation Results:")
    print(f"Red Wins: {stats['red_wins']} ({stats['red_wins']/num_games:.1%})")
    print(f"Yellow Wins: {stats['yellow_wins']} ({stats['yellow_wins']/num_games:.1%})")
    print(f"Draws: {stats['draws']} ({stats['draws']/num_games:.1%})")
    print(f"Win Rate: {stats['win_rate']:.2%}")
    
    # Calculate red win percentage among decisive games
    decisive_games = stats['red_wins'] + stats['yellow_wins']
    if decisive_games > 0:
        red_win_rate = stats['red_wins'] / decisive_games
        print(f"Red win rate (excl. draws): {red_win_rate:.1%}")
    
    return stats

def train_neural_network(model, states, policies, values, batch_size=32, epochs=10, validation_split=0.1):
    """Train the model with improved learning parameters"""
    # Ensure all inputs are float32 numpy arrays
    if not isinstance(states, np.ndarray):
        states = np.array(states, dtype=np.float32)
    if not isinstance(policies, np.ndarray):
        policies = np.array(policies, dtype=np.float32)
    if not isinstance(values, np.ndarray):
        values = np.array(values, dtype=np.float32)
    
    # Apply label smoothing to policy targets directly
    # Reduced smoothing to 0.03 to make policy distribution sharper
    smoothing = 0.03
    policies = (1 - smoothing) * policies + smoothing / policies.shape[-1]
    
    # Normalize the value targets to [-1, 1] range
    max_abs_value = np.max(np.abs(values)) + 1e-7  # Add epsilon to avoid division by zero
    values = values / max_abs_value

    # Create indices for train/validation split using standard numpy
    num_samples = len(states)
    indices = np.arange(num_samples)
    np.random.shuffle(indices)
    split_idx = int(num_samples * validation_split)
    val_indices = indices[:split_idx]
    train_indices = indices[split_idx:]

    # Split the data
    train_states = states[train_indices]
    train_policies = policies[train_indices]
    train_values = values[train_indices]
    
    val_states = states[val_indices]
    val_policies = policies[val_indices]
    val_values = values[val_indices]

    # Prepare callbacks
    log_dir = f"logs/fit/{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    callbacks = [
        tf.keras.callbacks.TensorBoard(
            log_dir=log_dir,
            histogram_freq=1,
            write_graph=True
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            min_delta=0.001
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath='best_model.keras',
            monitor='val_loss',
            save_best_only=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.7,  # Slower reduction
            patience=3,
            min_lr=1e-5
        )
    ]

    # Recompile the model with adjusted weights to strongly prefer policy learning
    model.compile(
        optimizer=tf.keras.optimizers.AdamW(
            learning_rate=0.0005,  # Reduced learning rate
            weight_decay=0.0003    # Reduced weight decay to help policy learn sharper distribution
        ),
        loss={
            'policy_output': 'categorical_crossentropy',
            'value_output': 'mean_squared_error'
        },
        loss_weights={
            'policy_output': 2.0,  # Significantly increased policy weight
            'value_output': 1.0
        },
        metrics={
            'policy_output': ['accuracy'],
            'value_output': ['mse']
        }
    )
    
    # Train in smaller batches if the dataset is large
    batch_size = min(batch_size, len(train_states) // 10 + 1)
    print(f"Using batch size: {batch_size}")

    # Train the model with standard fit
    history = model.fit(
        train_states,
        {'policy_output': train_policies, 'value_output': train_values},
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(
            val_states,
            {'policy_output': val_policies, 'value_output': val_values}
        ),
        callbacks=callbacks,
        verbose=1
    )

    return history

def main():
    num_iterations = 10
    games_per_iteration = 200  # Increased from 100 to get more training data
    training_log = []
    
    # Define evaluation games count
    evaluation_games = 50  # This matches the default in evaluate_model()

    # If model exists, start from the last iteration
    start_iteration = 0
    for i in range(num_iterations - 1, 0, -1):
        model_file = f'connect4_model_iteration_{i}.keras'
        if os.path.exists(model_file):
            start_iteration = i
            break

    for iteration in range(start_iteration, num_iterations):
        print(f"\nStarting iteration {iteration + 1}/{num_iterations}")
        
        # Load or create model
        model_file = f'connect4_model_iteration_{iteration}.keras'
        if os.path.exists(model_file):
            try:
                # First try to load the model normally
                model = tf.keras.models.load_model(model_file)
            except (ValueError, NotImplementedError) as e:
                # If we get any error, create a fresh model and load weights
                print(f"Error loading model: {str(e)}")
                print("Creating fresh model and loading weights instead...")
                
                # Create a new model with the same architecture
                fresh_model = Connect4NN(6, 7).model
                
                # Try to load weights
                try:
                    fresh_model.load_weights(model_file)
                    model = fresh_model
                except:
                    print("Could not load weights, using fresh model")
                    model = fresh_model
            print(f"Loaded existing model from iteration {iteration}")
            
            # Ensure the model is compiled properly
            model.compile(
                optimizer=tf.keras.optimizers.AdamW(learning_rate=0.0005),
                loss={
                    'policy_output': tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.03),  # Reduced from 0.05
                    'value_output': 'mean_squared_error'
                },
                loss_weights={'policy_output': 2.0, 'value_output': 1.0},  # Increased policy weight from 1.5
                metrics={'policy_output': ['accuracy'], 'value_output': ['mse']}
            )
        else:
            # Create a new model
            model = Connect4NN(6, 7).model
            print("Created new model")
            
        # Ensure model has built predict function
        dummy_input = np.zeros((1, 6, 7, 3), dtype=np.float32)
        _ = model.predict(dummy_input)  # Force building of predict function
        
        print("Model summary:")
        model.summary()

        # Generate training data using parallel self-play
        print("Generating training data...")
        # Increase games for first iteration to build initial dataset
        current_games = games_per_iteration * 2 if iteration == 0 else games_per_iteration
        training_data, game_stats = generate_training_data(model, current_games)
        print(f"Completed {current_games} games: {game_stats}")
        
        # Calculate game statistics for logging
        total_games = sum(game_stats.values())
        red_win_rate = game_stats["red_wins"] / total_games if total_games > 0 else 0
        yellow_win_rate = game_stats["yellow_wins"] / total_games if total_games > 0 else 0
        draw_rate = game_stats["draws"] / total_games if total_games > 0 else 0
        
        print(f"Game stats: Red wins: {red_win_rate:.1%}, Yellow wins: {yellow_win_rate:.1%}, Draws: {draw_rate:.1%}")
        
        # Check for policy collapse - if one side is winning >95% of games
        if red_win_rate > 0.95 or yellow_win_rate > 0.95:
            print("WARNING: Possible policy collapse detected! Increasing exploration parameters.")
            # Regenerate data with more exploration
            training_data, game_stats = generate_training_data(model, current_games)
        
        states, policies, values = prepare_training_data(training_data)
        
        # Calculate distribution and statistics of values for debugging
        print(f"Value distribution: Min={values.min():.2f}, Max={values.max():.2f}, Mean={values.mean():.2f}, Std={values.std():.2f}")
        print(f"Policy distribution entropy (should be >0): {-np.sum(policies * np.log(policies + 1e-10), axis=1).mean():.4f}")

        # Train the model
        print("Training model...")
        # Increase epochs for all iterations to ensure better learning
        epochs = 20 if iteration == 0 else 15  # Increased from 15/10
        history = train_neural_network(model, states, policies, values, batch_size=32, epochs=epochs)

        # Evaluate the model
        print("Evaluating model...")
        eval_results = evaluate_model(model, num_games=evaluation_games)

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

        # Save the model with a descriptive name
        output_model_name = f'connect4_engine_v{iteration + 2}.keras'
        model.save(output_model_name)
        print(f"Model saved as {output_model_name}")
        
        # Also save with iteration naming for backwards compatibility
        model.save(f'connect4_model_iteration_{iteration + 1}.keras')
        
        # Save training log
        with open('training_log.json', 'w') as f:
            json.dump(training_log, f, indent=2)

        print(f"\nIteration {iteration + 1} complete")
        print(f"Training metrics: {iteration_log['training_metrics']}")
        print(f"Evaluation results: {iteration_log['evaluation']}")
        
        # Break early if we achieve good balance (optional)
        if (0.35 <= eval_results["red_wins"]/evaluation_games <= 0.65 and 
            0.35 <= eval_results["yellow_wins"]/evaluation_games <= 0.65 and
            eval_results["draws"] > 0):
            print("Model has achieved good balance of red/yellow wins and draws!")

if __name__ == "__main__":
    # Set TensorFlow to use only the necessary GPU memory
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)
    
    main()
    
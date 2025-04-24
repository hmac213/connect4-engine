import numpy as np
from connect4 import Connect4
from model import Connect4NN
import os
import tensorflow as tf
from datetime import datetime
import json
import multiprocessing as mp
from functools import partial

# TODO: Create random state generation for varied starting states: pick a piece count randomly (0 - 42?) and make moves until the piece count is reached. If the game is over, undo the move and make another until a move cannot be made.

def play_game(model, tau=1.0, batch_size=32, initial_state=None):
    """Play a single game using direct policy predictions with batched inference"""
    game_history = []
    state = initial_state if initial_state is not None else Connect4()
    
    # Simplified random start (20% chance)
    if initial_state is None and np.random.random() < 0.2:
        state.random_start()

    # Pre-calculate temperature decay factors
    move_count = 0
    min_temp = 0.7
    
    # Simplified value tracking (only last 3 positions)
    last_values = [0, 0, 0]  # Initialize with neutral values
    value_idx = 0
    
    while not state.game_over():
        # Faster temperature calculation
        current_tau = tau * max(min_temp, 1.0 - (move_count / 35))
        
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
        
        # Simplified noise calculation
        noise_alpha = 0.3 if move_count < 20 else 0.5
        noise = np.random.dirichlet([noise_alpha] * len(valid_moves))
        
        # Simplified strength-based noise adjustment
        noise_weight = 0.35 if (state.turn == 1 and avg_value > 0.3) or \
                              (state.turn == 2 and avg_value < -0.3) else 0.25
        
        # Vectorized policy modification
        valid_policy = np.full(7, -np.inf)
        valid_policy[valid_moves] = (1 - noise_weight) * policy[valid_moves] + noise_weight * noise
        
        # Faster policy normalization
        valid_policy = np.exp(valid_policy / current_tau)
        valid_policy /= valid_policy.sum()

        # Choose move
        col = np.random.choice(7, p=valid_policy) if current_tau > 0 else valid_moves[np.argmax([valid_policy[m] for m in valid_moves])]

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
            policy_probs = policy_probs / policy_probs.sum()
        else:
            # Uniform distribution if no valid moves (shouldn't happen in normal play)
            policy_probs = np.ones(7) / 7
            
        training_data.append((state, policy_probs, value))
        
        # Add mirrored position with proper policy mirroring
        flipped_state = np.flip(state, axis=1)
        flipped_policy = np.flip(policy_probs)
        training_data.append((flipped_state, flipped_policy, value))
    
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

    return np.array(states), np.array(policies), np.array(values).reshape(-1, 1)

def evaluate_model(model, num_games=50):
    """Evaluate the model by playing games against itself with batched predictions"""
    results = []
    games_in_progress = []
    max_moves_per_game = 42  # Maximum possible moves in Connect4
    move_counter = {}  # Track moves per game
    
    # Start all games
    for _ in range(num_games):
        state = Connect4()
        if np.random.rand() < 0.2:  # 20% chance of random start
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
                    
                    # Make deterministic move during evaluation
                    valid_moves = [i for i in range(7) if state.is_valid_move(i)]
                    if not valid_moves:
                        results.append(3)  # Draw if no valid moves
                        continue
                        
                    # Filter policy for only valid moves
                    valid_policy = np.array([policy[i] if i in valid_moves else -np.inf for i in range(7)])
                    col = np.argmax(valid_policy)
                    
                    # Make move
                    player_turn = state.turn
                    try:
                        new_state = state.place_piece(col, player_turn)
                        
                        if new_state.game_over():
                            results.append(new_state.get_result())
                            if len(results) % 10 == 0:
                                print(f"Evaluated {len(results)}/{num_games} games")
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
    print(f"Red Wins: {stats['red_wins']}")
    print(f"Yellow Wins: {stats['yellow_wins']}")
    print(f"Draws: {stats['draws']}")
    print(f"Win Rate: {stats['win_rate']:.2%}")
    
    return stats

def train_neural_network(model, states, policies, values, batch_size=256, epochs=10, validation_split=0.1):
    """Train the model with improved learning parameters"""
    # Create TensorBoard callback with memory profiling
    log_dir = f"logs/fit/{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir,
        histogram_freq=1,
        write_graph=True,
        write_images=True,
        profile_batch='500,520'
    )

    # Create early stopping callback with more patience
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        min_delta=0.001,  # Smaller delta for finer convergence
        mode='min'
    )

    # Create model checkpoint callback
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath='best_model.keras',
        monitor='val_loss',
        save_best_only=True,
        save_weights_only=False,
        mode='min'
    )

    # Create learning rate scheduler with cosine decay
    initial_learning_rate = 1e-3  # Higher initial learning rate
    warmup_epochs = 2
    
    def cosine_decay_with_warmup(epoch):
        if epoch < warmup_epochs:
            return initial_learning_rate * ((epoch + 1) / warmup_epochs)
        else:
            decay_epochs = epochs - warmup_epochs
            epoch_in_decay = epoch - warmup_epochs
            cosine_decay = 0.5 * (1 + np.cos(np.pi * epoch_in_decay / decay_epochs))
            return initial_learning_rate * max(0.1, cosine_decay)  # Don't let LR go below 10% of initial
    
    lr_scheduler = tf.keras.callbacks.LearningRateScheduler(cosine_decay_with_warmup)

    # Convert inputs to float32
    states = tf.cast(states, tf.float32)
    policies = tf.cast(policies, tf.float32)
    values = tf.cast(values, tf.float32)

    # Add label smoothing to policy targets with curriculum
    base_smoothing = 0.1
    curriculum_steps = 5
    
    def get_smoothing_for_epoch(epoch):
        # Gradually reduce smoothing as training progresses
        progress = min(1.0, epoch / curriculum_steps)
        return base_smoothing * (1.0 - 0.5 * progress)
    
    # Normalize the value targets to [-1, 1] range
    values = values / np.abs(values).max()

    # Calculate split indices with stratification
    num_samples = len(states)
    num_validation = int(num_samples * validation_split)
    
    # Stratify based on game outcomes
    unique_values = np.unique(values)
    train_indices = []
    val_indices = []
    
    for value in unique_values:
        value_indices = np.where(values == value)[0]
        np.random.shuffle(value_indices)
        n_val = int(len(value_indices) * validation_split)
        val_indices.extend(value_indices[:n_val])
        train_indices.extend(value_indices[n_val:])
    
    # Shuffle the final indices
    np.random.shuffle(train_indices)
    np.random.shuffle(val_indices)

    # Split the data
    train_states = tf.gather(states, train_indices)
    train_policies = tf.gather(policies, train_indices)
    train_values = tf.gather(values, train_indices)
    
    val_states = tf.gather(states, val_indices)
    val_policies = tf.gather(policies, val_indices)
    val_values = tf.gather(values, val_indices)

    # Create data augmentation layer with curriculum
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.GaussianNoise(0.01)  # Reduced noise
    ])

    # Custom training step with curriculum learning
    class CustomTraining(tf.keras.callbacks.Callback):
        def on_epoch_begin(self, epoch, logs=None):
            # Update label smoothing
            smoothing = get_smoothing_for_epoch(epoch)
            self.model.compile(
                optimizer=tf.keras.optimizers.Adam(
                    learning_rate=cosine_decay_with_warmup(epoch),
                    beta_1=0.9,
                    beta_2=0.999,
                    epsilon=1e-7
                ),
                loss={
                    'policy_output': tf.keras.losses.CategoricalCrossentropy(label_smoothing=smoothing),
                    'value_output': 'mean_squared_error'
                },
                loss_weights={
                    'policy_output': 1.0,  # Equal weighting
                    'value_output': 1.0
                },
                metrics={
                    'policy_output': ['accuracy'],
                    'value_output': ['mse']
                }
            )

    # Prepare datasets with larger buffer
    train_dataset = tf.data.Dataset.from_tensor_slices(
        (train_states, {'policy_output': train_policies, 'value_output': train_values})
    )
    train_dataset = train_dataset.shuffle(buffer_size=50000)  # Larger shuffle buffer
    train_dataset = train_dataset.batch(batch_size)
    train_dataset = train_dataset.map(lambda x, y: (data_augmentation(x, training=True), y))
    train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)

    val_dataset = tf.data.Dataset.from_tensor_slices(
        (val_states, {'policy_output': val_policies, 'value_output': val_values})
    )
    val_dataset = val_dataset.batch(batch_size)
    val_dataset = val_dataset.prefetch(tf.data.AUTOTUNE)

    # Train the model
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs,
        callbacks=[
            tensorboard_callback,
            early_stopping,
            checkpoint_callback,
            lr_scheduler,
            CustomTraining()
        ],
        verbose=1
    )

    return history

def main():
    num_iterations = 10
    games_per_iteration = 100  # Reduced from original
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

        # Generate training data using parallel self-play
        print("Generating training data...")
        training_data, game_stats = generate_training_data(model, games_per_iteration)
        print(f"Completed {games_per_iteration} games: {game_stats}")
        
        states, policies, values = prepare_training_data(training_data)

        # Train the model
        print("Training model...")
        history = train_neural_network(model, states, policies, values, batch_size=32, epochs=10)

        # Evaluate the model
        print("Evaluating model...")
        eval_results = evaluate_model(model)

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
    # Set TensorFlow to use only the necessary GPU memory
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)
    
    main()
    
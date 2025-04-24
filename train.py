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
    # Modified temperature parameters for better exploration/exploitation balance
    min_temp = 0.3  # Reduced from 0.5 for sharper decisions late-game
    
    # Simplified value tracking (only last 3 positions)
    last_values = [0, 0, 0]  # Initialize with neutral values
    value_idx = 0
    
    while not state.game_over():
        # Faster temperature decay to encourage more deterministic policy decisions
        current_tau = tau * max(min_temp, 1.0 - (move_count / 20))  # Faster decay (30 → 20)
        
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
        
        # Value tracking with circular buffer
        last_values[value_idx] = value
        value_idx = (value_idx + 1) % 3
        avg_value = sum(last_values) / 3

        # Get valid moves
        valid_moves = [i for i in range(7) if state.is_valid_move(i)]
        
        # Random move with reduced probability to ensure better policy learning
        if np.random.random() < 0.02 and move_count < 4:  # Significantly reduced from 0.03 and 5 moves
            col = np.random.choice(valid_moves)
            player_turn = state.turn
            state = state.place_piece(col, player_turn)
            game_history.append((state.get_board(), policy, player_turn))
            move_count += 1
            continue
        
        # More strategic Dirichlet noise - stronger early, minimal late
        noise_alpha = 0.3 if move_count < 10 else 0.8  # Higher alpha = less noise later
        noise = np.random.dirichlet([noise_alpha] * len(valid_moves))
        
        # Apply Dirichlet noise with decreasing impact as game progresses
        noise_weight = max(0.05, 0.25 - move_count * 0.01)  # Gradually reduce noise influence
        
        # Vectorized policy processing
        valid_policy = np.full(7, -np.inf)
        valid_policy[valid_moves] = (1 - noise_weight) * policy[valid_moves] + noise_weight * noise
        
        # Apply temperature scaling
        valid_policy = np.exp(valid_policy / current_tau)
        valid_policy /= valid_policy.sum()

        # Progressive switch to deterministic play as game advances
        # Use stochastic policy early, gradually becoming more deterministic
        if move_count < 8 or np.random.random() < 0.1:  # Reduced from 10 moves
            col = np.random.choice(7, p=valid_policy)
        else:
            # Select the move with highest probability
            col = valid_moves[np.argmax([valid_policy[m] for m in valid_moves])]

        # Make move
        player_turn = state.turn
        state = state.place_piece(col, player_turn)
        game_history.append((state.get_board(), policy, player_turn))
        move_count += 1

    return game_history, state.get_result()

def process_game_result(game_history, outcome):
    """Process a single game's history into training data with proper value assignment"""
    training_data = []
    
    # Calculate the effective game length
    game_length = len(game_history)
    
    # Determine if the game ended in a win
    is_win = outcome in (1, 2)
    
    for idx, (board_state, policy, player) in enumerate(game_history):
        # Calculate position value based on outcome
        if outcome == 1:  # Red wins
            base_value = 1 if player == 1 else -1
        elif outcome == 2:  # Yellow wins
            base_value = -1 if player == 1 else 1
        else:  # Draw
            base_value = 0
            
        # Calculate position in the game (0 to 1)
        progress = idx / max(1, game_length - 1)
        
        # Apply temporal difference with stronger signals for decisive games
        # Winning/losing positions closer to the end have more accurate values
        moves_to_end = game_length - idx
        
        # Sharper value signal for wins/losses, smoother for draws
        if is_win:
            # Apply stronger discount for winning games (more confident predictions)
            discount_factor = 0.95  # Reduced from 0.99 for sharper signals
            value = base_value * (discount_factor ** moves_to_end)
            
            # Enhance values for moves closer to the end
            if moves_to_end < 10:
                # Amplify value for final sequences of moves
                value_modifier = 1.0 + (0.2 * (10 - moves_to_end) / 10)
                value = value * value_modifier
        else:
            # For draws, use a lower base value to discourage the model from playing for draws
            value = base_value * 0.7  # Reduce draw value to encourage decisive play
        
        # Clip value to valid range
        value = np.clip(value, -1.0, 1.0)
        
        # Process input state
        red_channel = (board_state == 1).astype(float)
        yellow_channel = (board_state == 2).astype(float)
        current_player_channel = np.ones_like(board_state) * (1 if player == 1 else 0)
        state = np.stack([red_channel, yellow_channel, current_player_channel], axis=-1)
        
        # Process policy - create proper probability distribution
        valid_moves = [i for i in range(7) if policy[i] > -np.inf]
        
        if valid_moves:
            # Normalize policy over valid moves
            policy_probs = np.zeros_like(policy)
            policy_probs[valid_moves] = policy[valid_moves]
            policy_probs = np.exp(policy_probs)
            
            # For winning moves close to the end, amplify the policy signal
            if is_win and moves_to_end < 8:
                winning_player = 1 if outcome == 1 else 2
                is_winner_move = (player == winning_player)
                
                if is_winner_move:
                    # Sharpen the policy distribution for winning sequences
                    temperature = max(0.3, 1.0 - (8 - moves_to_end) * 0.1)
                    policy_probs = np.power(policy_probs, 1/temperature)
            
            # Normalize to a proper distribution
            policy_probs = policy_probs / policy_probs.sum()
        else:
            # Fallback to uniform distribution (shouldn't normally happen)
            policy_probs = np.ones(7) / 7
            
        training_data.append((state, policy_probs, value))
        
        # Add mirrored position for data augmentation
        flipped_state = np.flip(state, axis=1)
        flipped_policy = np.flip(policy_probs)
        training_data.append((flipped_state, flipped_policy, value))
        
        # Add noise augmentation with controlled magnitude
        if np.random.random() < 0.1:  # Reduced from 0.15 to focus on actual game states
            # Copy the state and add minimal noise to prevent overfitting
            noise_state = state.copy()
            noise_state += np.random.normal(0, 0.02, noise_state.shape)  # Reduced noise
            noise_state = np.clip(noise_state, 0, 1)
            
            # Add less noise to policy for critical positions
            policy_noise_magnitude = 0.03 if moves_to_end < 8 else 0.05
            noise_policy = policy_probs * (1 + np.random.normal(0, policy_noise_magnitude, policy_probs.shape))
            noise_policy = noise_policy / noise_policy.sum()
            
            # Add minimal noise to value
            value_noise_magnitude = 0.03 if is_win and moves_to_end < 5 else 0.05
            noise_value = value + np.random.normal(0, value_noise_magnitude)
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
    """Train the model with improved learning parameters focusing on policy learning"""
    # Ensure all inputs are float32 numpy arrays
    if not isinstance(states, np.ndarray):
        states = np.array(states, dtype=np.float32)
    if not isinstance(policies, np.ndarray):
        policies = np.array(policies, dtype=np.float32)
    if not isinstance(values, np.ndarray):
        values = np.array(values, dtype=np.float32)
    
    # Policy sharpening - apply minimal label smoothing to maintain policy peaks
    smoothing = 0.01  # Reduced from 0.03 to make policy distribution sharper
    policies = (1 - smoothing) * policies + smoothing / policies.shape[-1]
    
    # Value normalization with soft clipping
    values = np.clip(values, -1.0, 1.0)

    # Create stratified validation split to ensure balanced training
    num_samples = len(states)
    indices = np.arange(num_samples)
    np.random.shuffle(indices)
    
    # Split into train/val sets with balanced sample distribution
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

    # Print dataset statistics for debugging
    print(f"Training on {len(train_states)} samples, validating on {len(val_states)} samples")
    print(f"Policy entropy (avg): {-np.sum(train_policies * np.log(train_policies + 1e-10), axis=1).mean():.4f}")
    print(f"Value distribution: mean={train_values.mean():.4f}, std={train_values.std():.4f}")

    # Configure callbacks with improved monitoring
    log_dir = f"logs/fit/{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    callbacks = [
        tf.keras.callbacks.TensorBoard(
            log_dir=log_dir,
            histogram_freq=1,
            write_graph=True,
            profile_batch=0  # Disable profiling to reduce overhead
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_policy_output_loss',  # Focus on policy improvement
            patience=5,
            restore_best_weights=True,
            min_delta=0.0005  # Reduced threshold to stop earlier
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath='best_model.keras',
            monitor='val_policy_output_loss',  # Focus on policy improvement
            save_best_only=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_policy_output_loss',  # Focus on policy improvement
            factor=0.5,  # Stronger reduction for more decisive learning rate changes
            patience=2,  # Reduced patience for faster adaptation
            min_lr=5e-6,
            verbose=1
        )
    ]

    # Recompile the model with stronger focus on policy learning
    model.compile(
        optimizer=tf.keras.optimizers.AdamW(
            learning_rate=0.0005,  # Keep learning rate moderate
            weight_decay=0.0001    # Further reduced weight decay for policy learning
        ),
        loss={
            'policy_output': 'categorical_crossentropy',
            'value_output': 'mean_squared_error'
        },
        loss_weights={
            'policy_output': 2.0,  # Even higher weight for policy learning
            'value_output': 1.0
        },
        metrics={
            'policy_output': ['accuracy', tf.keras.metrics.CategoricalCrossentropy(name='policy_ce')],
            'value_output': ['mse']
        }
    )
    
    # Adjust batch size based on dataset size
    if len(train_states) > 100000:
        batch_size = max(128, batch_size)  # Larger batch size for larger datasets
    elif len(train_states) < 10000:
        batch_size = min(32, batch_size)   # Smaller batch size for smaller datasets
    
    print(f"Using batch size: {batch_size}")

    # Train with class weights to focus on non-uniform policies
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
    games_per_iteration = 200
    training_log = []
    
    # Define evaluation games count
    evaluation_games = 100  # Increased from 50 for more reliable evaluation
    
    # Try to load training log if it exists
    if os.path.exists('training_log.json'):
        try:
            with open('training_log.json', 'r') as f:
                training_log = json.load(f)
            print(f"Loaded training log with {len(training_log)} previous iterations")
        except Exception as e:
            print(f"Error loading training log: {e}")
            training_log = []

    # If model exists, start from the last iteration
    start_iteration = 0
    for i in range(num_iterations - 1, -1, -1):
        model_file = f'connect4_model_iteration_{i}.keras'
        if os.path.exists(model_file):
            start_iteration = i
            print(f"Found existing model at iteration {i}")
            break
            
    print(f"Starting training from iteration {start_iteration}")

    for iteration in range(start_iteration, num_iterations):
        print(f"\n{'='*50}")
        print(f"STARTING ITERATION {iteration + 1}/{num_iterations}")
        print(f"{'='*50}")
        
        # Load or create model
        model_file = f'connect4_model_iteration_{iteration}.keras'
        best_model_file = 'best_model.keras'
        
        # First try to load the best model if it exists and we're continuing training
        if iteration > 0 and os.path.exists(best_model_file):
            try:
                model = tf.keras.models.load_model(best_model_file)
                print(f"Loaded best model from previous iteration")
            except Exception as e:
                print(f"Error loading best model: {e}")
                # Fall back to iteration model
                if os.path.exists(model_file):
                    model = tf.keras.models.load_model(model_file)
                    print(f"Loaded existing model from iteration {iteration}")
                else:
                    model = Connect4NN(6, 7).model
                    print("Created new model")
        elif os.path.exists(model_file):
            try:
                model = tf.keras.models.load_model(model_file)
                print(f"Loaded existing model from iteration {iteration}")
            except Exception as e:
                print(f"Error loading model: {e}")
                model = Connect4NN(6, 7).model
                print("Created new model after loading error")
        else:
            # Create a new model
            model = Connect4NN(6, 7).model
            print("Created new model")
            
        # Ensure model has built predict function with warm-up
        print("Warming up model...")
        dummy_input = np.zeros((1, 6, 7, 3), dtype=np.float32)
        _ = model.predict(dummy_input, verbose=0)
        
        # Adjust game count based on iteration and previous results
        if iteration == 0:
            # Start with more games to build initial training set
            current_games = games_per_iteration * 2
        elif iteration < 3:
            # Use standard number in early iterations
            current_games = games_per_iteration
        else:
            # Scale up as iterations progress
            current_games = int(games_per_iteration * (1 + iteration * 0.1))
            
        print(f"Playing {current_games} self-play games for training data...")
        
        # Generate training data using self-play
        training_data, game_stats = generate_training_data(model, current_games)
        
        # Calculate game statistics for monitoring
        total_games = sum(game_stats.values())
        red_win_rate = game_stats["red_wins"] / total_games if total_games > 0 else 0
        yellow_win_rate = game_stats["yellow_wins"] / total_games if total_games > 0 else 0
        draw_rate = game_stats["draws"] / total_games if total_games > 0 else 0
        
        print(f"Self-play results: Red wins: {red_win_rate:.1%}, Yellow wins: {yellow_win_rate:.1%}, Draws: {draw_rate:.1%}")
        
        # Check for policy collapse or imbalance
        if red_win_rate > 0.9 or yellow_win_rate > 0.9:
            print("WARNING: Severe policy imbalance detected! Increasing exploration and regenerating data.")
            # Regenerate data with more exploration by using higher temperature
            training_data, game_stats = generate_training_data(model, current_games)
            print(f"Regenerated data with results: Red: {game_stats['red_wins']/total_games:.1%}, Yellow: {game_stats['yellow_wins']/total_games:.1%}")
        
        # Prepare the training data
        states, policies, values = prepare_training_data(training_data)
        
        # Print data statistics for monitoring
        print(f"Training data prepared: {len(states)} positions")
        print(f"Value stats: Min={values.min():.2f}, Max={values.max():.2f}, Mean={values.mean():.2f}")
        
        # Check if we have policy collapse (all uniform policies)
        policy_entropy = -np.sum(policies * np.log(policies + 1e-10), axis=1).mean()
        print(f"Policy entropy: {policy_entropy:.4f} (should be >0.5 and <1.9)")
        
        if policy_entropy < 0.1:
            print("WARNING: Near-uniform policy detected! Adjusting training parameters.")
            # Add some targeted noise to policies to break uniform distribution
            noise = np.random.normal(0, 0.1, policies.shape)
            policies = policies * (1 + noise)
            policies = policies / policies.sum(axis=1, keepdims=True)

        # Train the model with appropriate epoch count based on iteration
        print("Training model...")
        if iteration == 0:
            epochs = 25  # More epochs in first iteration
        else:
            epochs = 15  # Standard epochs for subsequent iterations
            
        history = train_neural_network(
            model, states, policies, values, 
            batch_size=64,  # Increased from 32
            epochs=epochs
        )

        # Evaluate the model more thoroughly
        print("Evaluating model...")
        eval_results = evaluate_model(model, num_games=evaluation_games)

        # Log iteration results
        iteration_log = {
            "iteration": iteration + 1,
            "training_games": current_games,
            "self_play_stats": game_stats,
            "evaluation": eval_results,
            "training_metrics": {
                "policy_loss": float(history.history['policy_output_loss'][-1]),
                "value_loss": float(history.history['value_output_loss'][-1]),
                "policy_accuracy": float(history.history['policy_output_accuracy'][-1]),
                "policy_entropy": float(policy_entropy)
            },
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        training_log.append(iteration_log)

        # Save the model with version number
        output_model_name = f'connect4_engine_v{iteration + 1}.keras'
        model.save(output_model_name)
        print(f"Model saved as {output_model_name}")
        
        # Save with iteration naming for training continuity
        model.save(f'connect4_model_iteration_{iteration + 1}.keras')
        
        # Save training log
        with open('training_log.json', 'w') as f:
            json.dump(training_log, f, indent=2)

        print(f"\nIteration {iteration + 1} complete")
        print(f"Training metrics: {iteration_log['training_metrics']}")
        print(f"Evaluation results: Red wins: {eval_results['red_wins']}, Yellow wins: {eval_results['yellow_wins']}, Draws: {eval_results['draws']}")
        
        # Early stopping if we've achieved good performance
        if (iteration >= 3 and
            0.35 <= eval_results["red_wins"]/evaluation_games <= 0.65 and 
            0.35 <= eval_results["yellow_wins"]/evaluation_games <= 0.65 and
            eval_results["draws"]/evaluation_games <= 0.3):
            print("Model has achieved good balance with decisive play! Early stopping.")
            break

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
    
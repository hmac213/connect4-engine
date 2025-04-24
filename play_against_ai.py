import numpy as np
import os
import tensorflow as tf
from connect4 import Connect4
import time

# Enable unsafe deserialization for Lambda layers
tf.keras.config.enable_unsafe_deserialization()

# Define custom Lambda layer for casting to float32
def cast_to_float32(x):
    return tf.cast(x, dtype='float32')

# Register the function
tf.keras.utils.get_custom_objects().update({'cast_to_float32': cast_to_float32})

class Connect4CLI:
    def __init__(self, model_path="saved_models/connect4_engine_v1.keras", difficulty="medium"):
        self.game = Connect4()
        self.difficulty = difficulty  # "easy", "medium", "hard"
        
        try:
            # First attempt: Try importing the model directly
            from model import Connect4NN
            print("Creating fresh Connect4NN model...")
            self.model = Connect4NN(6, 7).model
            
            # If model path exists, try loading weights
            if os.path.exists(model_path):
                print(f"Loading weights from {model_path}...")
                try:
                    self.model.load_weights(model_path)
                    print("Successfully loaded weights!")
                except Exception as e:
                    print(f"Warning: Could not load weights directly: {e}")
                    print("Attempting to load full model for weight extraction...")
                    try:
                        # Try to load the full model with unsafe deserialization
                        custom_objects = {
                            'tf': tf,
                            'cast_to_float32': cast_to_float32
                        }
                        temp_model = tf.keras.models.load_model(
                            model_path, 
                            custom_objects=custom_objects, 
                            safe_mode=False,
                            compile=False  # Don't compile to avoid optimizer issues
                        )
                        
                        # Extract weights from each layer
                        for i, layer in enumerate(temp_model.layers):
                            try:
                                if i < len(self.model.layers):
                                    weights = layer.get_weights()
                                    if weights:
                                        self.model.layers[i].set_weights(weights)
                            except Exception as e:
                                print(f"Warning: Could not transfer weights for layer {i}: {e}")
                        
                        print("Successfully transferred weights from loaded model!")
                    except Exception as e:
                        print(f"Failed to load model for weight extraction: {e}")
                        print("Using untrained model - game will be very easy!")
            else:
                print(f"Model path {model_path} does not exist, using untrained model.")
        
        except Exception as e:
            print(f"Error during model initialization: {e}")
            print("Using a placeholder model - AI will play randomly.")
            # Create a minimal functional model as fallback
            input_layer = tf.keras.layers.Input(shape=(6, 7, 3))
            x = tf.keras.layers.Flatten()(input_layer)
            policy_head = tf.keras.layers.Dense(7, activation='softmax', name='policy_output')(x)
            value_head = tf.keras.layers.Dense(1, activation='tanh', name='value_output')(x)
            self.model = tf.keras.models.Model(inputs=input_layer, outputs=[policy_head, value_head])
        
        # Ensure model is compiled
        self.model.compile(
            optimizer='adam',
            loss={
                'policy_output': 'categorical_crossentropy',
                'value_output': 'mean_squared_error'
            }
        )
        
        # Ensure model has built predict function
        print("Building model prediction function...")
        dummy_input = np.zeros((1, 6, 7, 3), dtype=np.float32)
        _ = self.model.predict(dummy_input, verbose=0)
        print("Model ready!")
    
    def display_board(self):
        """Display the board in a human-readable format"""
        print("\n  1 2 3 4 5 6 7")  # Column numbers
        print(" ---------------")
        
        board = self.game.get_board()
        for row in range(5, -1, -1):  # Start from top row (5) to bottom row (0)
            print("|", end="")
            for col in range(7):
                if board[row][col] == 0:
                    print(" .", end="")
                elif board[row][col] == 1:
                    print(" R", end="")
                else:
                    print(" Y", end="")
            print(" |")
        print(" ---------------")
    
    def get_ai_move(self):
        """Get AI's move using the model with improved selection logic"""
        board = self.game.get_board()
        red_channel = (board == 1).astype(np.float32)
        yellow_channel = (board == 2).astype(np.float32)
        current_player_channel = np.ones_like(board, dtype=np.float32) * (1 if self.game.turn == 1 else 0)
        state_input = np.stack([red_channel, yellow_channel, current_player_channel], axis=-1)
        state_input = np.expand_dims(state_input, axis=0)
        
        print("\nAI is analyzing the position...")
        
        # Get model prediction
        policy, value = self.model.predict(state_input, verbose=0)
        policy = policy[0]
        value = value[0][0]  # Extract scalar value
        
        # Show AI's evaluation of the position
        player_perspective = value if self.game.turn == 1 else -value
        confidence = abs(player_perspective)
        position_eval = "AI strongly expects to win" if player_perspective > 0.7 else \
                        "AI likely to win" if player_perspective > 0.3 else \
                        "AI has slight advantage" if player_perspective > 0.1 else \
                        "Even position" if abs(player_perspective) <= 0.1 else \
                        "Human has slight advantage" if player_perspective > -0.3 else \
                        "Human likely to win" if player_perspective > -0.7 else "Human strongly expected to win"
        
        print(f"AI evaluation: {position_eval} (confidence: {confidence:.2f})")
        
        # Filter for valid moves only
        valid_moves = [i for i in range(7) if self.game.is_valid_move(i)]
        if not valid_moves:
            return None  # No valid moves
        
        # Simulate each valid move to find critical moves (wins or block opponent wins)
        for move in valid_moves:
            # Check if this move is a win
            test_game = self.game.copy()
            test_game.place_piece(move, test_game.turn)
            if test_game.check_win(self.game.turn):
                print("AI found winning move!")
                col = move
                time.sleep(1)  # Pause to let user see this message
                return col
        
        # Check if opponent would win on their next move
        opponent = 1 if self.game.turn == 2 else 2
        for move in valid_moves:
            test_game = self.game.copy()
            test_game.place_piece(move, self.game.turn)
            for opp_move in range(7):
                if test_game.is_valid_move(opp_move):
                    test_game2 = test_game.copy()
                    test_game2.place_piece(opp_move, opponent)
                    if test_game2.check_win(opponent):
                        # Opponent would win after this move, block it
                        test_game3 = self.game.copy()
                        test_game3.place_piece(opp_move, self.game.turn)
                        if test_game3.check_win(self.game.turn):
                            # We win by playing the opponent's winning move
                            print("AI found immediate winning move!")
                            time.sleep(1)  # Pause to let user see this message
                            return opp_move
                        else:
                            # We need to block this move
                            print("AI is blocking your winning move!")
                            time.sleep(1)  # Pause to let user see this message
                            return opp_move
        
        # Convert policy to probabilities for valid moves only with proper temperature
        # Use lower temperature (more deterministic) when the model is confident
        temperature = 0.3 if confidence > 0.7 else 0.5
        valid_policy = np.array([policy[i] if i in valid_moves else -np.inf for i in range(7)])
        
        # Display top 3 moves and their probabilities
        move_probs = [(i, valid_policy[i]) for i in range(7) if i in valid_moves]
        move_probs.sort(key=lambda x: x[1], reverse=True)
        print("AI's top moves:")
        for i, (move, prob) in enumerate(move_probs[:3]):
            print(f"  Column {move+1}: {prob:.2%}")
            
        # Apply temperature scaling
        valid_policy = np.exp(valid_policy / temperature)
        valid_policy = valid_policy / valid_policy.sum()
        
        # Deterministic in critical positions, more randomness in even positions
        if confidence > 0.7:  # Strong position, play best move
            col = np.argmax(valid_policy)
            print("Playing best move (high confidence)")
        elif confidence > 0.3:  # Clear advantage, mostly play best move
            if np.random.random() < 0.9:
                col = np.argmax(valid_policy)
                print("Playing best move (medium confidence)")
            else:
                col = np.random.choice(7, p=valid_policy)
                print("Playing weighted random move (medium confidence)")
        else:  # Unclear position, explore policy
            col = np.random.choice(7, p=valid_policy)
            print("Playing weighted random move (low confidence)")
        
        # Avoid moves that fill a column and allow easy win above
        if 0 <= col <= 6 and self.game.is_valid_move(col):
            # Check if this move enables a win in the column above
            test_game = self.game.copy()
            test_game.place_piece(col, test_game.turn)
            
            # If there's still room in the column
            if col < 7 and test_game.is_valid_move(col):
                next_test = test_game.copy()
                next_test.place_piece(col, opponent)
                if next_test.check_win(opponent):
                    print("AI avoided trap move!")
                    # Choose a different move
                    valid_policy[col] = 0
                    if valid_policy.sum() > 0:
                        valid_policy = valid_policy / valid_policy.sum()
                        col = np.random.choice(7, p=valid_policy)
        
        # Show final decision
        print(f"AI chose column {col+1}")
        
        # Pause to let user read the AI's reasoning
        if self.difficulty != "easy":  # Only pause for medium/hard modes
            input("Press Enter to see the AI make its move...")
        
        return col
    
    def get_human_move(self):
        """Get human player's move"""
        while True:
            try:
                move = input("Your move (1-7): ")
                col = int(move) - 1  # Convert to 0-indexed
                
                if 0 <= col <= 6 and self.game.is_valid_move(col):
                    return col
                else:
                    print("Invalid move. Please choose a column between 1-7 that isn't full.")
            except ValueError:
                print("Please enter a number between 1 and 7.")
    
    def play_game(self):
        """Main game loop"""
        # Clear screen and show welcome message
        os.system('cls' if os.name == 'nt' else 'clear')
        print("=== Connect 4 vs AI ===")
        print("You can play as Red (R) or Yellow (Y).")
        
        # Let player choose color
        while True:
            color_choice = input("Choose your color (R/Y): ").upper()
            if color_choice in ['R', 'Y']:
                break
            print("Invalid choice. Please enter R or Y.")
        
        # Let player choose difficulty
        while True:
            difficulty_choice = input("Choose difficulty (easy/medium/hard): ").lower()
            if difficulty_choice in ['easy', 'medium', 'hard']:
                self.difficulty = difficulty_choice
                break
            print("Invalid choice. Please enter easy, medium, or hard.")
        
        human_player = 1 if color_choice == 'R' else 2
        ai_player = 2 if human_player == 1 else 1
        
        print(f"\nYou are {'Red' if human_player == 1 else 'Yellow'}!")
        print(f"AI is {'Red' if ai_player == 1 else 'Yellow'} (Difficulty: {self.difficulty.capitalize()})")
        print("\nMake your move by entering a column number (1-7).")
        print("Starting game in 3 seconds...")
        time.sleep(3)
        
        # Game loop
        game_over = False
        move_history = []
        
        if self.game.turn != human_player:
            # AI makes first move
            os.system('cls' if os.name == 'nt' else 'clear')
            print(f"=== Connect 4 vs AI === (Difficulty: {self.difficulty.capitalize()})")
            print(f"You: {'Red' if human_player == 1 else 'Yellow'} | AI: {'Red' if ai_player == 1 else 'Yellow'}")
            self.display_board()
            print("\nAI is thinking...")
            
            # Adjust AI behavior based on difficulty
            if self.difficulty == "easy":
                # Easy mode: 50% random moves
                if np.random.random() < 0.5:
                    valid_moves = [i for i in range(7) if self.game.is_valid_move(i)]
                    ai_move = np.random.choice(valid_moves)
                    print("AI chose a random move")
                else:
                    ai_move = self.get_ai_move()
            elif self.difficulty == "medium":
                # Medium: normal AI but with more randomness
                ai_move = self.get_ai_move()
            else:
                # Hard: full strength AI
                ai_move = self.get_ai_move()
            
            self.game.place_piece(ai_move, ai_player)
            move_history.append(ai_move + 1)
            print(f"AI placed in column {ai_move + 1}")
            time.sleep(1)
        
        while not game_over:
            os.system('cls' if os.name == 'nt' else 'clear')
            print(f"=== Connect 4 vs AI === (Difficulty: {self.difficulty.capitalize()})")
            print(f"You: {'Red' if human_player == 1 else 'Yellow'} | AI: {'Red' if ai_player == 1 else 'Yellow'}")
            
            # Show move history
            if move_history:
                print(f"Move history: {' '.join(map(str, move_history))}")
            
            self.display_board()
            
            # Check if game is over
            if self.game.game_over():
                game_over = True
                break
            
            # Human turn
            if self.game.turn == human_player:
                human_move = self.get_human_move()
                self.game.place_piece(human_move, human_player)
                move_history.append(human_move + 1)
                
                # Check if game is over after human move
                if self.game.game_over():
                    os.system('cls' if os.name == 'nt' else 'clear')
                    print(f"=== Connect 4 vs AI === (Difficulty: {self.difficulty.capitalize()})")
                    print(f"You: {'Red' if human_player == 1 else 'Yellow'} | AI: {'Red' if ai_player == 1 else 'Yellow'}")
                    print(f"Move history: {' '.join(map(str, move_history))}")
                    self.display_board()
                    game_over = True
                    break
                
                # AI turn - clear screen and show updated board first
                os.system('cls' if os.name == 'nt' else 'clear')
                print(f"=== Connect 4 vs AI === (Difficulty: {self.difficulty.capitalize()})")
                print(f"You: {'Red' if human_player == 1 else 'Yellow'} | AI: {'Red' if ai_player == 1 else 'Yellow'}")
                print(f"Move history: {' '.join(map(str, move_history))}")
                self.display_board()
                print("\nAI is thinking...")
                
                # Adjust AI behavior based on difficulty
                if self.difficulty == "easy":
                    # Easy mode: 50% random moves
                    if np.random.random() < 0.5:
                        valid_moves = [i for i in range(7) if self.game.is_valid_move(i)]
                        ai_move = np.random.choice(valid_moves)
                        print("AI chose a random move")
                    else:
                        ai_move = self.get_ai_move()
                elif self.difficulty == "medium":
                    # Medium: normal AI but with more randomness
                    ai_move = self.get_ai_move()
                else:
                    # Hard: full strength AI 
                    ai_move = self.get_ai_move()
                
                # Wait for user to confirm before making the move
                input("\nPress Enter to see AI's move...")
                
                self.game.place_piece(ai_move, ai_player)
                move_history.append(ai_move + 1)
                print(f"AI placed in column {ai_move + 1}")
                
                # Wait for user to confirm before continuing
                input("\nPress Enter to continue to your turn...")
        
        # Game over - show result
        os.system('cls' if os.name == 'nt' else 'clear')
        print(f"=== Connect 4 vs AI === GAME OVER")
        print(f"You: {'Red' if human_player == 1 else 'Yellow'} | AI: {'Red' if ai_player == 1 else 'Yellow'}")
        print(f"Move history: {' '.join(map(str, move_history))}")
        self.display_board()
        
        result = self.game.get_result()
        if result == human_player:
            print("🎉 Congratulations! You win! 🎉")
        elif result == ai_player:
            print("AI wins! Better luck next time.")
        else:
            print("It's a draw!")
        
        # Ask if player wants to play again
        play_again = input("Play again? (y/n): ").lower()
        if play_again == 'y':
            self.game = Connect4()  # Reset game
            self.play_game()  # Start new game
        else:
            print("Thanks for playing!")

if __name__ == "__main__":
    # Try to find the model
    model_path = "saved_models/connect4_engine_v1.keras"
    print("\n=== Connect4 AI Loading ===")
    
    if os.path.exists(model_path):
        print(f"Found model: {model_path}")
        print("Loading model... (this may take a few seconds)")
        game = Connect4CLI(model_path)
    else:
        print(f"Model not found at {model_path}")
        print("Using default model path or creating a new model.")
        game = Connect4CLI()
    
    # Pause to allow user to see loading messages
    print("\nModel loading complete. Press Enter to start the game...")
    input()
    
    # Start the game
    game.play_game() 
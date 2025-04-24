import numpy as np
import random

class Connect4:
    # 1 is red
    # 2 is yellow
    # 3 for results is draw
    def __init__(self):
        self.row_count = 6
        self.col_count = 7
        self.game_running = True
        self.turn = 1
        self.board = np.zeros((self.row_count, self.col_count))

    def get_board(self):
        return self.board

    def is_valid_move(self, col):
        # First check if column is in bounds
        if col < 0 or col >= self.col_count:
            return False
        # Then check if the top cell in the column is empty
        return self.board[self.row_count - 1][col] == 0
    
    def legal_moves(self):
        return [(col, self.turn) for col in range(self.col_count) if self.is_valid_move(col)]

    def check_win(self, color):
        for r in range(self.row_count):
            for c in range(self.col_count - 3):
                if self.board[r][c] == color and self.board[r][c + 1] == color and self.board[r][c + 2] == color and self.board[r][c + 3] == color:
                    # print(f"horizontal win starting at row: {r}, col: {c}")
                    return True
                
        for r in range(self.row_count - 3):
            for c in range(self.col_count):
                if self.board[r][c] == color and self.board[r + 1][c] == color and self.board[r + 2][c] == color and self.board[r + 3][c] == color:
                    # print(f"vertical win starting at row: {r}, col: {c}")
                    return True
                
        for r in range(self.row_count - 3):
            for c in range(self.col_count - 3):
                if self.board[r][c] == color and self.board[r + 1][c + 1] == color and self.board[r + 2][c + 2] == color and self.board[r + 3][c + 3] == color:
                    # print(f"up diagonal win starting at row: {r}, col: {c}")
                    return True
                
        for r in range(3, self.row_count):
            for c in range(self.col_count - 3):
                if self.board[r][c] == color and self.board[r - 1][c + 1] == color and self.board[r - 2][c + 2] == color and self.board[r - 3][c + 3] == color:
                    # print(f"down diagonal win starting at row: {r}, col: {c}")
                    return True
        
        return False
    
    def game_over(self):
        for color in [1, 2]:
            if self.check_win(color):
                # print(f"the winner is {color}")
                return True

        if not self.legal_moves():
            # print("there are no legal moves")
            return True

        return False

    
    def get_result(self):
        if self.game_over():
            if self.check_win(1):
                # print("the game is over: Player 1 wins")
                return 1
            elif self.check_win(2):
                # print("the game is over: Player 2 wins")
                return 2
            else:
                # print("the game is over: Player 3 wins")
                return 3
        else:
            return 0

    def place_piece(self, col, color):
        if self.is_valid_move(col):
            for row_num in range(self.row_count):
                if self.board[row_num][col] == 0:
                    self.board[row_num][col] = color
                    break

        self.turn = 2 if color == 1 else 1
        return self
    
    def print_board(self):
        print(self.board)

    def copy(self):
        new_copy = Connect4()
        new_copy.board = np.copy(self.board)
        new_copy.turn = self.turn
        return new_copy
    
    def revert_state(self, buffer_game):
        self.board = np.copy(buffer_game.board)
        self.turn = buffer_game.turn

    def random_start(self):
        num_turns = np.random.randint(1, 43)

        for _ in range(num_turns):
            open_moves = self.legal_moves()

            move_made = False

            while open_moves:
                move = random.choice(open_moves)
                open_moves.remove(move)
                buffer_game = self.copy()
                self.place_piece(move[0], move[1])

                if not self.game_over():
                    move_made = True
                    break
                else:
                    self.revert_state(buffer_game)

            if not move_made:
                break

        if self.game_over():
            self.revert_state(buffer_game)

        return self
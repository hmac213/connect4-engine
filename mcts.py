import numpy as np

class MCTSNode:
    def __init__(self, state, parent = None):
        self.state = state
        self.parent = parent
        self.children = {}
        self.visits = 0
        self.value_sum = 0
        self.prior = 0

    @property
    def value(self):
        return self.value_sum / self.visits if self.visits > 0 else 0
    def is_fully_expanded(self):
        return len(self.children) == len(self.state.legal_moves())
    
    def best_child(self, c_param=1.4):
        # UCB1 calculation: value + exploration term
        choices_weights = []
        for move, child in self.children.items():
            if child.visits > 0:
                uct_value = child.value + c_param * np.sqrt((2 * np.log(self.visits) / child.visits))
            else:
                uct_value = c_param * np.sqrt(2 * np.log(self.visits))

            choices_weights.append(uct_value)
            # print(f"Move: {move}, Visits: {child.visits}, UCT Value: {uct_value}")
        
        best_move = np.max(choices_weights)
        # print(choices_weights)
        best_move_indices = np.where(choices_weights == best_move)[0]
        # print(best_move_indices)
        if len(best_move_indices) > 1:
            best_move_index = np.random.choice(best_move_indices)
        else:
            best_move_index = best_move_indices[0]

        best_move = list(self.children.keys())[best_move_index]
        # print(f"Best move selected: {best_move} with UCT value {choices_weights[best_move_index]}")
        return self.children[best_move]
    
    def expand(self):
        legal_moves = self.state.legal_moves()
        for move in legal_moves:
            if move[0] not in self.children:
                new_state = self.state.copy()
                new_state.place_piece(move[0], move[1])
                child_node = MCTSNode(new_state, self)
                self.children[move[0]] = child_node
                return child_node
        return None

    
    def backpropagate(self, value):
        self.visits += 1
        self.value_sum += value
        if self.parent:
            self.parent.backpropagate(-value)

class MCTS:
    def __init__(self, neural_net, simulations):
        self.neural_net = neural_net
        self.simulations = simulations

    def search(self, state, temp=1):
        root = MCTSNode(state)
        policy_array = np.zeros(7)  # Initialize the policy array for 7 possible moves

        for _ in range(self.simulations):
            node = root
            while node.is_fully_expanded() and node.children:
                # temperature enabled for training
                node = node.best_child()

            if not node.is_fully_expanded():
                expanded_node = node.expand()
                if expanded_node is None:
                    # print(f"reached empty node from position \n {node.state.get_board()}")
                    break
                node = expanded_node

            # Neural Network Evaluation
            state_input = np.array(node.state.get_board()).reshape((1, 6, 7, 1))
            policy, value = self.neural_net.predict(state_input)

            for move, p in enumerate(policy[0]):
                if move not in node.children:
                    child_state = node.state.copy()
                    child_state.place_piece(move, node.state.turn)
                    child_node = MCTSNode(child_state, parent=node)
                    child_node.prior = p
                    node.children[move] = child_node

            node.backpropagate(value[0][0])

        # Build policy based on visit counts
        for move, child in root.children.items():
            policy_array[move] = child.visits  # Store visit count of each child

        # Normalize the policy to create a probability distribution
        policy_array = np.power(policy_array, 1 / temp)
        policy_array = policy_array / np.sum(policy_array)

        # print(policy_array)
        return policy_array  # Return the policy distribution

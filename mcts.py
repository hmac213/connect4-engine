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
        choices_weights = []
        for move, child in self.children.items():
            if child.visits > 0:
                uct_value = child.value + c_param * np.sqrt((2 * np.log(self.visits) / child.visits))
            else:
                uct_value = c_param * np.sqrt(2 * np.log(self.visits))
            choices_weights.append(uct_value)
        
        best_move_index = np.argmax(choices_weights)
        return self.children[list(self.children.keys())[best_move_index]]
    
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
        self.batch_size = 32  # Batch size for neural network predictions

    def prepare_input(self, state):
        """Prepare 3-channel input for the neural network"""
        board = state.get_board()
        red_channel = (board == 1).astype(float)
        yellow_channel = (board == 2).astype(float)
        current_player_channel = np.ones_like(board) * (1 if state.turn == 1 else 0)
        return np.stack([red_channel, yellow_channel, current_player_channel], axis=-1)

    def batch_predict(self, states):
        """Batch process multiple states through the neural network"""
        inputs = np.array([self.prepare_input(state) for state in states])
        policies, values = self.neural_net.predict(inputs, batch_size=self.batch_size)
        return policies, values

    def search(self, state, temp=1):
        root = MCTSNode(state)
        policy_array = np.zeros(7)

        # Initial evaluation of the root node
        state_input = self.prepare_input(root.state)
        state_input = np.expand_dims(state_input, axis=0)
        root_policy, root_value = self.neural_net.predict(state_input)
        
        # Initialize children with prior probabilities
        for move, p in enumerate(root_policy[0]):
            if move in state.legal_moves():
                child_state = state.copy()
                child_state.place_piece(move, state.turn)
                child_node = MCTSNode(child_state, root)
                child_node.prior = p
                root.children[move] = child_node

        # Batch processing for simulations
        batch_states = []
        batch_nodes = []
        
        for _ in range(self.simulations):
            node = root
            path = []
            
            # Selection
            while node.is_fully_expanded() and node.children:
                node = node.best_child()
                path.append(node)
            
            if not node.is_fully_expanded():
                expanded_node = node.expand()
                if expanded_node is not None:
                    node = expanded_node
                    path.append(node)
            
            if node not in batch_nodes:
                batch_states.append(node.state)
                batch_nodes.append(node)
            
            # Process batch when it's full or at the end of simulations
            if len(batch_states) == self.batch_size or _ == self.simulations - 1:
                if batch_states:
                    policies, values = self.batch_predict(batch_states)
                    
                    for i, (policy, value) in enumerate(zip(policies, values)):
                        current_node = batch_nodes[i]
                        
                        # Update children with prior probabilities
                        for move, p in enumerate(policy):
                            if move not in current_node.children:
                                child_state = current_node.state.copy()
                                child_state.place_piece(move, current_node.state.turn)
                                child_node = MCTSNode(child_state, current_node)
                                child_node.prior = p
                                current_node.children[move] = child_node
                        
                        # Backpropagate the value
                        current_node.backpropagate(value[0])
                
                batch_states = []
                batch_nodes = []

        # Build final policy
        for move, child in root.children.items():
            policy_array[move] = child.visits

        # Normalize the policy
        policy_array = np.power(policy_array, 1 / temp)
        policy_array = policy_array / np.sum(policy_array)

        return policy_array

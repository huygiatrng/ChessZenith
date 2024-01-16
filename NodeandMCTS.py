import numpy as np
import torch
import math


def safe_normalize(policy, valid_moves_mask):
    sum_policy = np.sum(policy)
    if sum_policy > 0:
        return policy / sum_policy
    else:
        num_valid_moves = np.sum(valid_moves_mask)
        if num_valid_moves > 0:
            return valid_moves_mask / num_valid_moves
        else:
            # Handle the case when there are no valid moves
            # This should typically not happen in a standard game state
            return valid_moves_mask  # or raise an exception


class Node:
    def __init__(self, game, args, state, move_index_mapping, parent=None, action_taken=None, prior=0, visit_count=0):
        self.game = game
        self.args = args
        self.state = state
        self.move_index_mapping = move_index_mapping
        self.parent = parent
        self.action_taken = action_taken
        self.prior = prior

        self.children = []
        self.visit_count = visit_count
        self.value_sum = 0
        self.sqrt_visit_count = 0  # Store the square root of visit_count

    def is_fully_expanded(self):
        return bool(self.children)

    def select(self):
        best_child = None
        best_ucb = -float('inf')
        sqrt_parent_visit_count = math.sqrt(self.visit_count)  # Pre-compute this value

        for child in self.children:
            ucb = self._get_ucb(child, sqrt_parent_visit_count)
            if ucb > best_ucb:
                best_child = child
                best_ucb = ucb

        return best_child

    def _get_ucb(self, child, sqrt_parent_visit_count):
        q_value = 0 if child.visit_count == 0 else 1 - ((child.value_sum / child.visit_count) + 1) / 2
        return q_value + self.args['C'] * (sqrt_parent_visit_count / (child.visit_count + 1)) * child.prior

    def expand(self, policy):
        # Pre-filter actions with non-zero probability
        for action, prob in [(action, prob) for action, prob in enumerate(policy) if prob > 0]:
            uci_move = self.game.index_to_move(action, self.move_index_mapping)
            if uci_move:
                child_state = self.game.get_next_state(self.state.copy(), uci_move)
                child_state = self.game.change_perspective(child_state, player=-1)
                self.children.append(
                    Node(self.game, self.args, child_state, self.move_index_mapping, self, action, prob))

    def backpropagate(self, value):
        # Convert to iterative backpropagation
        node = self
        while node:
            node.value_sum += value
            node.visit_count += 1
            node.sqrt_visit_count = math.sqrt(node.visit_count)  # Update the sqrt value
            value = self.game.get_opponent_value(value)
            node = node.parent


class MCTS:
    def __init__(self, game, args, model):
        self.game = game
        self.args = args
        self.model = model
        self.cache = {}  # Cache for storing state evaluations

    @torch.no_grad()
    def search(self, state):
        policy, _ = self.evaluate_state(state)
        valid_moves_mask, move_index_mapping = self.game.get_valid_moves(state)
        policy *= valid_moves_mask

        # Use safe normalization
        policy = safe_normalize(policy, valid_moves_mask)

        root = Node(self.game, self.args, state, move_index_mapping, visit_count=1)
        root.expand(policy)

        batch_states = []
        nodes_to_expand = []
        batch_size = self.args['batch_to_evaluate_size']

        for _ in range(self.args['num_searches']):
            node = root
            while node.is_fully_expanded():
                node = node.select()

            if not node.children:  # Node needs to be expanded
                batch_states.append(node.state)
                nodes_to_expand.append(node)
                if len(batch_states) == batch_size:
                    self.evaluate_and_expand_batch(nodes_to_expand, batch_states)
                    batch_states = []
                    nodes_to_expand = []

        if batch_states:  # Process any remaining states
            self.evaluate_and_expand_batch(nodes_to_expand, batch_states)

        # Vectorized action probability calculation
        action_indices = [child.action_taken for child in root.children]
        visit_counts = np.array([child.visit_count for child in root.children])
        action_probs = np.zeros(self.game.action_size)
        action_probs[action_indices] = visit_counts
        action_probs /= action_probs.sum()

        return action_probs

    def evaluate_state(self, state):
        # Use FEN string as a cache key
        state_key = self.game.get_fen(state)
        if state_key in self.cache:
            return self.cache[state_key]

        # Convert state to tensor and perform model evaluation
        state_tensor = torch.tensor(self.game.get_encoded_state(state), device=self.model.device).unsqueeze(0)
        policy, value = self.model(state_tensor)
        policy = torch.softmax(policy, axis=1).squeeze(0).cpu().numpy()

        # Apply Dirichlet noise for root node
        if self.game.is_root_node(state):
            policy = (1 - self.args['dirichlet_epsilon']) * policy + \
                     self.args['dirichlet_epsilon'] * np.random.dirichlet(
                [self.args['dirichlet_alpha']] * self.game.action_size)

        # Cache the results
        self.cache[state_key] = (policy, value.item())

        return policy, value.item()

    def evaluate_and_expand_batch(self, nodes, states):
        batch_policy, batch_value = self.batch_evaluate_states(states)
        for node, policy, value in zip(nodes, batch_policy, batch_value):
            valid_moves, _ = self.game.get_valid_moves(node.state)
            policy *= valid_moves
            policy = safe_normalize(policy, valid_moves)
            # policy /= np.sum(policy)
            node.expand(policy)
            node.backpropagate(value)

    def batch_evaluate_states(self, states):
        # Convert list of states to batch tensor
        state_tensors = [torch.tensor(self.game.get_encoded_state(state), device=self.model.device) for state in states]
        batch_tensor = torch.stack(state_tensors)

        # Perform batch evaluation
        with torch.no_grad():
            batch_policy, batch_value = self.model(batch_tensor)
            batch_policy = torch.softmax(batch_policy, axis=1).cpu().numpy()
            batch_value = batch_value.cpu().numpy()

        return batch_policy, batch_value

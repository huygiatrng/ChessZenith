import numpy as np
import chess
import random

class ChessGame:
    def __init__(self):
        self.row_count = 8
        self.column_count = 8
        self.action_size = 4672
        self.board = chess.Board()

    def __repr__(self):
        return "Chess"

    def get_initial_state(self):
        # Initialize to the standard chess starting position
        self.board.reset()
        return self.board

    def index_to_move(self, index, move_index_mapping):
        # Use the provided dynamic mapping
        return {v: k for k, v in move_index_mapping.items()}.get(index, None)

    def get_next_state(self, state, action):
        # Execute the move on the board
        try:
            move = chess.Move.from_uci(action)
            if move in state.legal_moves:
                state.push(move)
        except ValueError:
            print("Invalid move format")
            pass  # Handle invalid move format
        return state

    def is_valid_move(self, state, move_uci):
        move = chess.Move.from_uci(move_uci)
        return move in state.legal_moves

    def get_valid_moves(self, state):
        valid_moves_mask = np.zeros(self.action_size, dtype=np.float32)
        move_index_mapping = {}
        index = 0

        for move in state.legal_moves:
            uci_move = move.uci()
            if uci_move not in move_index_mapping:
                move_index_mapping[uci_move] = index
                index += 1
            valid_moves_mask[move_index_mapping[uci_move]] = 1

        return valid_moves_mask, move_index_mapping

    def play_random_move(self, state):
        legal_moves = list(state.legal_moves)
        if legal_moves:
            state.push(random.choice(legal_moves))
        return state

    def check_win(self, state):
        # Check for checkmate, stalemate, or draw
        return state.is_checkmate(), state.is_stalemate() or state.is_insufficient_material() or state.can_claim_draw()

    def get_value_and_terminated(self, state, action):
        # Check for checkmate or standard draw conditions
        checkmate, draw = self.check_win(state)
        if checkmate:
            # Assuming the current player is the winner
            return 1, True
        if draw:
            return 0, True
        # 50-move rule
        if state.can_claim_fifty_moves():
            return 0, True
        # Threefold repetition
        if state.can_claim_threefold_repetition():
            return 0, True
        # Move limit (e.g., 100 moves, you can adjust this number)
        MOVE_LIMIT = 75
        if state.fullmove_number >= MOVE_LIMIT:
            return 0, True
        return 0, False

    def get_opponent(self, player):
        return not player

    def get_opponent_value(self, value):
        return -value

    def change_perspective(self, state, player):
        # This might involve flipping the board for the other player
        return state.mirror() if not player else state

    def get_encoded_state(self, state):
        # Create an empty 3D array with 12 layers (6 piece types x 2 players)
        encoded_state = np.zeros((12, 8, 8), dtype=np.float32)

        piece_map = state.piece_map()
        for position, piece in piece_map.items():
            row, col = divmod(position, 8)
            layer_index = piece.piece_type - 1 + (6 if piece.color == chess.BLACK else 0)
            encoded_state[layer_index, row, col] = 1

        return encoded_state

    def get_fen(self, state):
        return state.fen()

    def is_root_node(self, state):
        return state.fullmove_number == 1 and state.turn == chess.WHITE
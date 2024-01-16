import pandas as pd
import chess
import json
import os
from datetime import datetime

# Load the dataset
df = pd.read_csv(r'../real_game/games.csv')

# Constants
DATA_FOLDER = r'real_game\data'  # Update this to your data folder path


def board_to_matrix(board):
    """Converts a chess board to an 8x8 matrix representation."""
    matrix = []
    for i in range(8):
        row = []
        for j in range(8):
            piece = board.piece_at(chess.square(j, i))
            row.append(str(piece) if piece else ' ')
        matrix.append(row)
    return matrix


def san_to_uci(san_moves, fen=chess.STARTING_FEN):
    board = chess.Board(fen)
    uci_moves = []

    for san in san_moves:
        move = board.parse_san(san)
        uci_moves.append(move.uci())
        board.push(move)

    return uci_moves


def board_to_fen(board):
    """Converts a chess board to FEN representation."""
    return board.fen()


def san_moves_to_game_states(moves_san_list):
    """Convert a list of moves in SAN format into game states, ignoring move numbers and game results."""
    board = chess.Board()
    game_states = []

    # Filtering out result indicators
    result_indicators = ['1-0', '0-1', '1/2-1/2']
    moves_san_list = [move for move in moves_san_list if move not in result_indicators]

    for san_move in moves_san_list:
        if '.' in san_move:
            # Skip move numbers
            continue
        try:
            move = board.parse_san(san_move)
            # save the current board state and next move
            game_states.append({
                'board_state': board_to_fen(board),
                'chosen_move': move.uci()
            })
            board.push(move)
        except chess.InvalidMoveError:
            print(f"Invalid move encountered: {san_move}")
            break

    return game_states


def save_game_data(game_id, game_states, outcome):
    """Save the game data in JSON format."""
    game_data = {
        'game_id': game_id,
        'game_states': game_states,
        'outcome': outcome
    }

    outcome_folder = os.path.join(DATA_FOLDER, outcome)
    os.makedirs(outcome_folder, exist_ok=True)
    file_path = os.path.join(outcome_folder, f'{game_id}.json')

    with open(file_path, 'w') as file:
        json.dump(game_data, file, indent=4)

    print(f"Game data saved to {file_path}")


# Iterate through each game in the CSV
total_games = len(df)
for index, row in df.iterrows():
    game_id = row['id']
    moves_san_list = row['moves'].split()
    outcome = 'win' if row['winner'] == 'white' else 'lose' if row['winner'] == 'black' else 'draw'

    game_states = san_moves_to_game_states(moves_san_list)
    save_game_data(game_id, game_states, outcome)

    # Print progress
    print(f'Processed game {index + 1} of {total_games} (Game ID: {game_id})')

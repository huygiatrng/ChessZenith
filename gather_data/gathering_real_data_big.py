import pandas as pd
import chess
import json
import os
import random

def load_and_process_csv(file_path):
    # Load the dataset
    df = pd.read_csv(file_path)

    # Process the 'Result' column based on the last part of the 'AN' column
    def determine_result(an):
        if '1-0' in an:
            return 'win'
        elif '0-1' in an:
            return 'lose'
        elif '1/2-1/2' in an:
            return 'draw'
        else:
            return 'unknown'

    df['Result'] = df['AN'].apply(determine_result)
    return df

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
            board.push(move)

            game_states.append({
                'board_state': board_to_fen(board),
                'chosen_move': san_move
            })
        except chess.InvalidMoveError:
            print(f"Invalid move encountered: {san_move}")
            break

    return game_states

def save_game_data(game_id, game_states, outcome, data_folder):
    """Save the game data in JSON format."""
    game_data = {
        'game_id': game_id,
        'game_states': game_states,
        'outcome': outcome
    }

    outcome_folder = os.path.join(data_folder, outcome)
    os.makedirs(outcome_folder, exist_ok=True)
    file_path = os.path.join(outcome_folder, f'{game_id}.json')

    with open(file_path, 'w') as file:
        json.dump(game_data, file, indent=4)

    print(f"Game data saved to {file_path}")

def process_games(csv_file_path, data_folder):
    df = load_and_process_csv(csv_file_path)

    existing_game_ids = set()
    for index, row in df.iterrows():
        an = row['AN']

        # Skip games with special characters like {, }, or %
        if any(char in an for char in ['{', '}', '%']):
            continue

        # Generate unique game_id
        game_id = random.randint(100000000, 999999999)
        while game_id in existing_game_ids:
            game_id = random.randint(100000000, 999999999)
        existing_game_ids.add(game_id)

        # Convert SAN moves to game states
        moves_san_list = an.split()
        game_states = san_moves_to_game_states(moves_san_list)

        # Determine the outcome
        outcome = row['Result']

        # Save the game data
        save_game_data(game_id, game_states, outcome, data_folder)


# Example usage
csv_file_path = r'../big_real_game/chess_games.csv'
data_folder = r'big_real_game\data'
process_games(csv_file_path, data_folder)


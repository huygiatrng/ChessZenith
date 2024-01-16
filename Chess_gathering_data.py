import chess
import chess.svg
import pygame
import random
import os
import json
import os
from datetime import datetime

# Program to gathering chess data

# Constants
SIZE_OF_SQUARE = 80
SIZE_OF_BOARD = SIZE_OF_SQUARE * 8
MAX_STEPS = 300
DATA_FOLDER = r'mnt\data'
PIECES_FOLDER = "pieces"
NUMBER_OF_GAMES = 3000
FPS = 240

# Initialize Pygame for visualization
pygame.init()
screen = pygame.display.set_mode((SIZE_OF_BOARD, SIZE_OF_BOARD))
clock = pygame.time.Clock()


def load_pieces_images():
    """Load and resize chess pieces images to fit the squares."""
    pieces_images = {}
    for piece in ['b', 'w']:
        for type in ['pawn', 'knight', 'bishop', 'rook', 'queen', 'king']:
            image_path = os.path.join(PIECES_FOLDER, f'{piece}-{type}.png')
            key = f'{piece}-{type}'
            try:
                image = pygame.image.load(image_path)
                # Resize image to fit square size
                image = pygame.transform.scale(image, (SIZE_OF_SQUARE, SIZE_OF_SQUARE))
                pieces_images[key] = image
            except Exception as e:
                print(f"Failed to load image: {key}, Error: {e}")
    return pieces_images

pieces_images = load_pieces_images()


def draw_board(board):
    """Draw the board and pieces on the screen."""
    # Mapping from single-letter symbols to full names
    piece_name_map = {'r': 'rook', 'n': 'knight', 'b': 'bishop', 'q': 'queen', 'k': 'king', 'p': 'pawn'}

    for i in range(8):
        for j in range(8):
            rect = pygame.Rect(j * SIZE_OF_SQUARE, i * SIZE_OF_SQUARE, SIZE_OF_SQUARE, SIZE_OF_SQUARE)
            pygame.draw.rect(screen, (255, 255, 255) if (i + j) % 2 == 0 else (125, 125, 125), rect)
            piece = board.piece_at(chess.square(j, i))
            if piece:
                piece_symbol = str(piece).lower()  # Convert to lowercase
                piece_full_name = piece_name_map[piece_symbol]  # Get full name of the piece
                color_prefix = 'b' if piece.color == chess.BLACK else 'w'
                piece_key = f'{color_prefix}-{piece_full_name}'
                screen.blit(pieces_images[piece_key], rect)


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

def play_random_game():
    """Play a game with random moves and save data."""
    board = chess.Board()
    game_states = []

    for _ in range(MAX_STEPS):
        # Handle Pygame events to keep the window responsive
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return [], 'quit'  # Return empty list and 'quit' if window is closed

        if board.is_game_over():
            break

        moves = list(board.legal_moves)
        if not moves:
            break

        move = random.choice(moves)
        san_move = board.san(move)

        # Save the current state and the chosen move for policy network
        game_states.append({
            'board_state': board_to_fen(board.copy()),  # Use board_to_fen here
            'chosen_move': san_move
        })

        board.push(move)

        draw_board(board)
        pygame.display.flip()
        clock.tick(FPS)

    # Revised outcome determination
    outcome = 'draw'  # Default outcome
    if board.is_checkmate():
        # Let AI is white
        outcome = 'win' if board.turn == chess.BLACK else 'lose'
    elif board.is_stalemate() or board.is_insufficient_material() or \
         board.can_claim_draw() or board.is_seventyfive_moves() or \
         board.is_fivefold_repetition() or board.is_fifty_moves():
        outcome = 'draw'

    return game_states, outcome

def save_game_data(game_states, outcome):
    """Save the game data in JSON format."""
    game_id = datetime.now().strftime("%Y%m%d%H%M%S")
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


# Main loop for data gathering
for i in range(NUMBER_OF_GAMES):
    print(f"Game #{i} - ", end="")
    game_states, outcome = play_random_game()

    if outcome == 'quit':
        break  # Exit the loop if the Pygame window was closed
    save_game_data(game_states, outcome)

pygame.quit

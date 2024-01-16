import torch
import multiprocessing
from ChessGame import ChessGame
from ZenithModel import ZenithModel
from ResNet import ResNet



if __name__ == '__main__':
    chess_game = ChessGame()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNet(8, 256, chess_game.action_size, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)

    # Using multiprocessing module
    num_cores_multiprocessing = multiprocessing.cpu_count()
    ratio_workers = 0.6
    num_workers = int(num_cores_multiprocessing * ratio_workers)
    print(
        f"We have {num_cores_multiprocessing} cores. Use {ratio_workers * 100}%, which is {num_workers} for self-play.")

    args = {
        'C': 2,
        'checkpoints_dir': 'checkpoints',
        'num_searches': 150,
        'num_iterations': 10,
        'num_selfPlay_iterations': 500,
        'num_epochs_pretrain': 30,
        'num_epochs': 10,
        'batch_size': 128,
        'temperature': 1.25,
        'dirichlet_epsilon': 0.25,
        'dirichlet_alpha': 0.3,
        'num_workers': num_workers,
        'batch_to_evaluate_size': 256,
    }
    alphaZero = ZenithModel(model, optimizer, chess_game, args)
    # alphaZero.learn(resume=False)

    resume_training = False  # Set to True if resuming training
    model_path = 'checkpoints/model_1.pt'  # Path to the saved model file
    optimizer_path = 'checkpoints/optimizer_1.pt'  # Path to the saved optimizer file

    if resume_training:
        alphaZero.load_model(model_path, optimizer_path)
        alphaZero.learn(resume=True)
    else:
        # alphaZero.learn(resume=False)
        alphaZero.pretrain_and_learn("real_game/data")

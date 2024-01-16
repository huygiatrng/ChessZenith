import os
import json
import numpy as np
import torch
import torch.nn.functional as F
import random
from NodeandMCTS import MCTS
from tqdm import tqdm
import time
from multiprocessing import Pool


class ZenithModel:
    def __init__(self, model, optimizer, game, args):
        self.model = model
        self.optimizer = optimizer
        self.game = game
        self.args = args
        self.mcts = MCTS(game, args, model)

    def selfPlay(self, game_id=0):
        memory = []
        player = 1
        state = self.game.get_initial_state()
        move_count = 0
        total_time = 0

        while True:
            start_time = time.time()
            neutral_state = self.game.change_perspective(state, player)
            action_probs = self.mcts.search(neutral_state)

            _, move_index_mapping = self.game.get_valid_moves(state)

            memory.append((neutral_state, action_probs, player))
            action_index = np.random.choice(self.game.action_size, p=action_probs)
            action_uci = self.game.index_to_move(action_index, move_index_mapping)

            elapsed_time = time.time() - start_time
            total_time += elapsed_time
            move_count += 1

            if action_uci is not None:
                state = self.game.get_next_state(state, action_uci)
            else:
                pass

            value, is_terminal = self.game.get_value_and_terminated(state, action_uci)

            if is_terminal:
                avg_time_per_move = total_time / move_count
                # print(f"Game {game_id}: Total Moves = {move_count}, Avg Time/Move = {avg_time_per_move:.2f} seconds")
                return self.process_memory(memory, value, player)

            player = self.game.get_opponent(player)

    def process_memory(self, memory, value, player):
        returnMemory = []
        for hist_neutral_state, hist_action_probs, hist_player in memory:
            hist_outcome = value if hist_player == player else self.game.get_opponent_value(value)
            returnMemory.append((
                self.game.get_encoded_state(hist_neutral_state),
                hist_action_probs,
                hist_outcome
            ))
        return returnMemory

    def load_model(self, model_path, optimizer_path):
        self.model.load_state_dict(torch.load(model_path, map_location=self.model.device))
        self.optimizer.load_state_dict(torch.load(optimizer_path, map_location=self.model.device))

    def train(self, memory):
        random.shuffle(memory)
        for batchIdx in range(0, len(memory), self.args['batch_size']):
            sample = memory[batchIdx:min(len(memory) - 1, batchIdx + self.args[
                'batch_size'])]
            state, policy_targets, value_targets = zip(*sample)

            state, policy_targets, value_targets = np.array(state), np.array(policy_targets), np.array(
                value_targets).reshape(-1, 1)

            state = torch.tensor(state, dtype=torch.float32, device=self.model.device)
            policy_targets = torch.tensor(policy_targets, dtype=torch.float32, device=self.model.device)
            value_targets = torch.tensor(value_targets, dtype=torch.float32, device=self.model.device)

            out_policy, out_value = self.model(state)

            policy_loss = F.cross_entropy(out_policy, policy_targets)
            value_loss = F.mse_loss(out_value, value_targets)
            loss = policy_loss + value_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    def learn(self, memory=[], resume=False):
        if resume:
            self.model.train()
            with tqdm(total=self.args['num_epochs'], desc="Resuming Training Progress") as pbar:
                for _ in range(self.args['num_epochs']):
                    self.train(memory)
                    pbar.update(1)
        else:
            for iteration in range(self.args['num_iterations']):
                self.model.eval()
                with tqdm(total=self.args['num_selfPlay_iterations'], desc="Self Play Progress") as pbar:
                    with Pool(processes=self.args['num_workers']) as pool:
                        # Create a list of arguments for self-play games
                        game_args = [(i,) for i in range(self.args['num_selfPlay_iterations'])]

                        # Use imap_unordered for better progress tracking
                        for new_memory in pool.imap_unordered(self.selfPlay, game_args):
                            memory.extend(new_memory)
                            pbar.update(1)  # Update progress after each game is completed

                self.model.train()
                with tqdm(total=self.args['num_epochs'], desc="Training Progress") as pbar:
                    for _ in range(self.args['num_epochs']):
                        self.train(memory)
                        pbar.update(1)

                data_dir = self.args['checkpoints_dir']
                torch.save(self.model.state_dict(), f"{data_dir}/model_{iteration}.pt")
                torch.save(self.optimizer.state_dict(), f"{data_dir}/optimizer_{iteration}.pt")

    def pretrain_with_real_data(self, data_folder):
        total_games = self.count_total_games(data_folder)
        print(f"Found: {total_games} games")
        total_states = self.count_total_states(data_folder)
        memory = []

        with tqdm(total=total_states, desc="Pretraining Progress") as pbar:
            for outcome_label in ['win', 'lose', 'draw']:
                outcome_value = {'win': 1, 'lose': -1, 'draw': 0}[outcome_label]
                outcome_dir = os.path.join(data_folder, outcome_label)

                for filename in os.listdir(outcome_dir):
                    file_path = os.path.join(outcome_dir, filename)
                    with open(file_path, 'r') as file:
                        game_data = json.load(file)

                        for state_info in game_data['game_states']:
                            chosen_move = state_info['chosen_move']
                            board_state = state_info['board_state']
                            self.game.board.set_fen(board_state)
                            encoded_state = self.game.get_encoded_state(self.game.board)

                            if chosen_move and self.game.is_valid_move(self.game.board, chosen_move):
                                _, move_index_mapping = self.game.get_valid_moves(self.game.board)
                                action_probs = np.zeros(self.game.action_size)
                                move_index = move_index_mapping.get(chosen_move)
                                if move_index is not None:
                                    action_probs[move_index] = 1
                                memory.append((encoded_state, action_probs, outcome_value))
                            else:
                                print("invalid move detected.")

                            pbar.update(1)  # Update progress bar after processing each state

        self.model.train()
        with tqdm(total=self.args['num_epochs_pretrain'], desc="Training Real Data Progress") as pbar:
            for _ in range(self.args['num_epochs_pretrain']):
                self.train(memory)
                pbar.update(1)

        return memory

    def pretrain_and_learn(self, data_dir):
        self.learn(self.pretrain_with_real_data(data_dir))

    def count_total_states(self, data_folder):
        total_states = 0
        for outcome_label in ['win', 'lose', 'draw']:
            outcome_dir = os.path.join(data_folder, outcome_label)
            for filename in os.listdir(outcome_dir):
                file_path = os.path.join(outcome_dir, filename)
                with open(file_path, 'r') as file:
                    game_data = json.load(file)
                    total_states += len(game_data['game_states'])
        return total_states

    def count_total_games(self, data_dir):
        text_file_count = 0
        for root, dirs, files in os.walk(data_dir):
            for file in files:
                if file.endswith(".json"):
                    text_file_count += 1
        return text_file_count

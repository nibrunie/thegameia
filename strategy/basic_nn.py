# -*- coding: utf-8 -*-
import random
import argparse

# TensorFlow and tf.keras
import numpy as np
import tensorflow as tf
from tensorflow import keras

from basic_player import BasePlayer, thegameia_strategy, Action, ACTION_LIST
from game_simulator import Game
from utils import verbose_report

from strategy.nn_utils import stack_state_to_vector, value_to_one_hot
from strategy.medium_player import StarterPlayer, MediumPlayer
from strategy.random_player import RandomPlayer


def generate_one_card_training_data(num_games):
    """ generate training data using @p num_games parties simulation """
    training_data = []
    game = Game(RandomPlayer(), RandomPlayer())
    for i in range(num_games):
        # simulating a new game
        game.reset()
        player_id = 0
        while True:
            player = game.players[player_id]
            opponent = game.get_opponent(player_id)
            player_vector = stack_state_to_vector(player.stack_state)
            opponent_vector = stack_state_to_vector(opponent.stack_state)
            # generating input / expected for each card in hand
            for card in player.hand:
                is_valid = player.has_valid_play(opponent, card)
                input_vector = np.concatenate((player_vector, opponent_vector, value_to_one_hot(card.value)))
                output_vector = 1 if is_valid else 0
                training_data.append((
                    input_vector,
                    output_vector
                ))
            valid = game.play_one_turn(player_id)
            if not valid: break
            if game.players[player_id].win_condition():
                verbose_report("player {} has won".format(player_id))
                break
            # switch to next player
            player_id = 1 - player_id
    return training_data


@thegameia_strategy
class TrainablePlayer(StarterPlayer):
    def __init__(self):
        pass

    def build_model(self):
        print("building model")
        # input state is [4 stack values, One single hand card value]
        # output is [probability card is invalid, probability card is valid]
        self.model = keras.Sequential([
            keras.layers.Dense(16, activation=tf.nn.relu, input_shape=(60*5,)),
            keras.layers.Dense(16, activation=tf.nn.relu),
            keras.layers.Dense(16, activation=tf.nn.relu),
            keras.layers.Dense(2, activation=tf.nn.softmax),
        ])
        self.model.summary()

        self.model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

    def train_model(self, num_party=2000, epochs=5):
        print("generating training data")
        training_data = generate_one_card_training_data(num_party)
        print("training model")
        training_inputs = np.array([inputs for inputs, _ in training_data])
        training_outputs = np.array([outputs for _, outputs in training_data])
        print(training_inputs.shape, training_outputs.shape)
        self.model.fit(training_inputs, training_outputs, epochs=epochs)

    def evaluate_model(self, num_party=10):
        print("evaluating model")
        label_data = generate_one_card_training_data(num_party)
        label_inputs = np.array([inputs for inputs, _ in label_data])
        label_outputs = np.array( [outputs for _, outputs in label_data])
        test_loss, test_acc = self.model.evaluate(label_inputs, label_outputs)
        print('Test accuracy:', test_acc)


    def execute_model(self, player, opponent):
        # build state
        player_vector = stack_state_to_vector(player.stack_state)
        opponent_vector = stack_state_to_vector(opponent.stack_state)
        for card in player.hand:
            input_vector = np.concatenate((player_vector, opponent_vector, value_to_one_hot(card.value)))
            predictions = self.model.predict(np.array([input_vector]))
            prediction = np.argmax(predictions[0,:])
            is_valid = player.has_valid_play(opponent, card)
            print(card, predictions, prediction, is_valid)

def main_trainable_player():
    parser = argparse.ArgumentParser(description="NN based IA for the Game")
    parser.add_argument("--num", type=int, default=100, help="number of party to simulate for NN training")
    parser.add_argument("--epochs", type=int, default=5, help="number of epoch to run for training")
    parser.add_argument("--save-file", type=str, default=None, help="NN weights will be saved to this file")
    parser.add_argument("--load-file", type=str, default=None, help="NN weights will be stored from this file (bypass training)")
    parser.add_argument("--skip-training", action="store_const", default=False, const=True, help="skip training phase")
    args = parser.parse_args()

    # train and evaluate model
    nn_player = TrainablePlayer()
    nn_player.build_model()
    if args.load_file:
        nn_player.model.load_weights(args.load_file)

    if not args.skip_training:
        nn_player.train_model(args.num, args.epochs)

    if args.save_file:
        nn_player.model.save_weights(args.save_file)
        print("evaluating NN during one game")

        # execute model on one game
        game = Game(RandomPlayer(), RandomPlayer())
        # simulating a new game
        game.reset()
        player_id = 0
        while True:
            player = game.players[player_id]
            opponent = game.get_opponent(player_id)
            # evaluating model
            print("\nnew evaluation")
            player.display_state(str(player_id))
            nn_player.execute_model(player, opponent)

            #
            valid = game.play_one_turn(player_id)
            if not valid: break
            if game.players[player_id].win_condition():
                verbose_report("player {} has won".format(player_id))
                break
            # switch to next player
            player_id = 1 - player_id

    nn_player.evaluate_model()

if __name__ == "__main__":
    main_trainable_player()

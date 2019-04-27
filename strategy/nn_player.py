# -*- coding: utf-8 -*-
import random

# TensorFlow and tf.keras
import numpy
import tensorflow as tf
from tensorflow import keras

from basic_player import BasePlayer, thegameia_strategy
from game_simulator import Game
from utils import verbose_report

from medium_player import StarterPlayer
from random_player import RandomPlayer

def value_to_one_hot(value):
    vector = new_empty_vector()
    vector[value-1] = 1.0
    return vector

def new_empty_vector():
    return [0.0] * 60

def stack_state_to_vector(stack_state):
    return sum([value_to_one_hot(card.value) for card in stack_state], [])

def full_state_to_vector(state):
    """ Translate a list of [increasing stack top, decreasing stack top, hand cards]
        to the concatenation of 8 60-wide one hot vectors """
    # complementary vector completes hand to 6 cards
    complementary_vector = new_empty_vector() * (8 - len(state))
    return sum([value_to_one_hot(card.value) for card in state], []) + complementary_vector


class TrainablePlayer(StarterPlayer):
    def __init__(self):
        pass

    def build_model(self):
        print("building model")
        # input state is [4 stack values, 6 hand card value]
        # output is [6 probability of playing hand card]
        self.model = keras.Sequential([
            keras.layers.Dense(16, activation=tf.nn.relu, input_shape=(60*5,)),
            keras.layers.Dense(16, activation=tf.nn.relu),
            keras.layers.Dense(16, activation=tf.nn.relu),
            keras.layers.Dense(2, activation=tf.nn.softmax),
        ])

        self.model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

    def train_model(self, num_party=2000):
        print("generating training data")
        training_data = self.generate_training_data(num_party)
        print("training model")
        training_inputs = numpy.array([inputs for inputs, _ in training_data])
        training_outputs = numpy.array([outputs for _, outputs in training_data])
        print(training_inputs.shape, training_outputs.shape)
        self.model.fit(training_inputs, training_outputs, epochs=5)

    def evaluate_model(self, num_party=10):
        print("evaluating model")
        label_data = self.generate_training_data(num_party)
        label_inputs = numpy.array([inputs for inputs, _ in label_data])
        label_outputs = numpy.array( [outputs for _, outputs in label_data])
        test_loss, test_acc = self.model.evaluate(label_inputs, label_outputs)
        print('Test accuracy:', test_acc)


    def generate_training_data(self, num_games):
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
                    input_vector = numpy.array(player_vector + opponent_vector + value_to_one_hot(card.value))
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

    def execute_model(self, player, opponent):
        # build state
        player_vector = stack_state_to_vector(player.stack_state)
        opponent_vector = stack_state_to_vector(opponent.stack_state)
        for card in player.hand:
            input_vector = numpy.array(player_vector + opponent_vector + value_to_one_hot(card.value))
            predictions = self.model.predict(numpy.array([input_vector]))
            prediction = predictions[0][0] / predictions[0][1] < 1.0
            is_valid = player.has_valid_play(opponent, card)
            print(card, predictions, prediction, is_valid)


    def play_state(self, opponent):
        state = [
            self.increasing_list[-1].value,
            self.decreasing_list[-1].value,
            opponent.increasing_list[-1].value,
            opponent.decreasing_list[-1].value,
        ]
        # adding hand state
        state += [card.value for card in self.hand]

if __name__ == "__main__":
    # train and evaluate model
    nn_player = TrainablePlayer()
    nn_player.build_model()
    nn_player.train_model()
    nn_player.evaluate_model()
    if False:
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

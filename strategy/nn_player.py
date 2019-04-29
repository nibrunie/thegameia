# -*- coding: utf-8 -*-
import random
import argparse

# TensorFlow and tf.keras
import numpy as np
import tensorflow as tf
from tensorflow import keras

from basic_player import BasePlayer, thegameia_strategy
from game_simulator import Game
from utils import verbose_report

from medium_player import StarterPlayer
from random_player import RandomPlayer

def value_to_one_hot(value):
    vector = new_empty_vector()
    vector[value-1] = 1
    return vector

def new_empty_vector():
    return np.zeros(60)

def stack_state_to_vector(stack_state):
    assert len(stack_state) == 2
    vector = np.zeros(2 * 60)
    for index, card in enumerate(stack_state):
        vector[index * 60 + card.value - 1] = 1
    return vector

def full_state_to_vector(state):
    """ Translate a list of [increasing stack top, decreasing stack top, hand cards]
        to the concatenation of 8 60-wide one hot vectors """
    # complementary vector completes hand to 6 cards
    complementary_vector = new_empty_vector() * (8 - len(state))
    return sum([value_to_one_hot(card.value) for card in state], []) + complementary_vector

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


    def play_state(self, opponent):
        state = [
            self.increasing_list[-1].value,
            self.decreasing_list[-1].value,
            opponent.increasing_list[-1].value,
            opponent.decreasing_list[-1].value,
        ]
        # adding hand state
        state += [card.value for card in self.hand]

class KnowledgablePlayer(TrainablePlayer):
    def __init__(self):
        pass

	def get_state(self, opponent):
		player = self
        player_vector = stack_state_to_vector(player.stack_state)
        opponent_vector = stack_state_to_vector(opponent.stack_state)
        input_vector = np.concatenate(
			[player_vector, opponent_vector] + [value_to_one_hot(card.value) for card in player.hand] +
			[np.zeros(60)] * (6 - len(player.hand))
		)
        return input_vector

    def build_model(self):
        print("building model")
        # input state is [4 stack values, 6 hand card value]
        # output is [24 probabilities of playing one of the hand card on one of the stacks]
        self.model = keras.Sequential([
            keras.layers.Dense(16, activation=tf.nn.relu, input_shape=(60*10,)),
            keras.layers.Dense(16, activation=tf.nn.relu),
            keras.layers.Dense(16, activation=tf.nn.relu),
            keras.layers.Dense(24, activation=tf.nn.linear),
        ])
        self.model.summary()

        self.model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

	def evaluate_state(self, opponent):
		# we search to minize the increase of our increasing stack
		self_score = 60.0 + (self.decreasing_list[-1].value - self.increasing_list[-1].value) 
		self_factor = 3.0
		# we look for maximizing our opponent stack difference
		opp_score  = 60.0 + (opponent.increasing_list[-1].value - opponent.decreasing_list[-1].value) 
		opp_factor = 1.0
		return self_factor * self_score + opp_factor * opp_score 

    def train_model(self, num_party=100, epochs=5):
        print("generating training data")
        training_data = generate_one_card_training_data(num_party)
        print("training model")
        training_inputs = np.array([inputs for inputs, _ in training_data])
        training_outputs = np.array([outputs for _, outputs in training_data])
        print(training_inputs.shape, training_outputs.shape)
        self.model.fit(training_inputs, training_outputs, epochs=epochs)

		# now execute the q learning
		# learning parameters
		y = 0.95
		eps = 0.5
		decay_factor = 0.999
		r_avg_list = []
    	game = Game(self, RandomPlayer())
		opponent = game.get_opponent(0)
		for i in range(num_party):
			s = game.reset()
			eps *= decay_factor
			if i % (num_party / 10) == 0:
				print("Episode {} of {}".format(i + 1, num_episodes))
			r_sum = 0
			game_ended = False
			while not game_ended:
				current_state = self.get_state(opponent)
				target_vec = self.model.predict(current_state)[0] 
				if np.random.random() < eps:
					card_id = np.random.randint(0, 6)
					action_id = np.random.randint(0, 4)
				else:
					a = np.argmax(target_vec)
					card_id = int(a % 6)
					action_id = int(a / 6)
				player_action = Action(self.hand[card_id], ACTION_LIST[action_id])
				# should check if action is valid
				if self.is_action_valid(player_action, game.get_opponent(0))
					self.execute(player_action, game.get_opponent(0))
					reward = self.evaluate_state(opponent) 
				else:
					# recompense is zero (invalid play)
					reward = 0.0
					# random play
					possible_actions = sum([self.register_valid_actions(game.get_opponent(0)) for card in self.hand], [])
					if len(possible_actions) != 0:
						player_action = random.choice(possible_actions)
						self.execute(player_action, game.get_opponent(0))
					else:
						game_ended = True
				
				if game_ended:
					# reward malus
					reward -=
				else:

					if self.win_condition():
						# reward bonus  
						reward += 
						game_ended = True

					else:
						# play opponent turn (to get update next state)
						if not game.play_one_turn(1):
							# game stopped
							game_ended = True
						elif opponent.win_condition():
							# reward malus
							reward -= 
							game_ended = True

						next_state = self.get_state(opponent)
					
				target = reward + y * np.max(self.model.predict(next_state))
				target_vec[a] = target
				self.model.fit(current_state, target_vec.reshape(-1, 24), epochs=1, verbose=0)
				r_sum += reward
				
			r_avg_list.append(r_sum / 1000)

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
    parser = argparse.ArgumentParser(description="NN based IA for the Game")
    parser.add_argument("--num", type=int, default=100, help="number of party to simulate for NN training")
    parser.add_argument("--epochs", type=int, default=5, help="number of epoch to run for training")
    parser.add_argument("--save-file", type=str, default=None, help="NN weights will be saved to this file")
    parser.add_argument("--load-file", type=str, default=None, help="NN weights will be stored from this file (bypass training)")
    args = parser.parse_args()

    # train and evaluate model
    nn_player = TrainablePlayer()
    nn_player.build_model()
    if args.load_file:
        nn_player.model.load_weights(args.load_file)

    else:
        nn_player.train_model(args.num, args.epochs)
        if args.save_file:
            nn_player.model.save_weights(args.save_file)
    if True:
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

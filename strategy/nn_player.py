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

from medium_player import StarterPlayer
from random_player import RandomPlayer

def value_to_one_hot(value):
    """ covert index @p value to a one-hot vector of size 60 (card state)
        encoding index """
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
        super().__init__()
        # statistics
        self.play_count = 0
        self.valid_count = 0
        # internal storage for next action
        self.available_action = None
        # player Neural Network model structure
        self.model = None

    def build_model(self):
        """ Build player Neural Network model """
        print("building model")
        # input state is [4 stack values, 6 hand card value]
        # output is [24 probabilities of playing one of the hand card on one of the stacks]
        self.model = keras.Sequential([
            keras.layers.Dense(16, activation=tf.nn.relu, input_shape=(60*10,)),
            keras.layers.Dense(16, activation=tf.nn.relu),
            keras.layers.Dense(16, activation=tf.nn.relu),
            keras.layers.Dense(24 * 24, activation="linear"),
        ])
        self.model.summary()

        self.model.compile(optimizer='adam',
                      loss='mse',
                      metrics=['accuracy'])

    def get_state(self, opponent):
        """ build player state vector """
        player = self
        player_vector = stack_state_to_vector(player.stack_state)
        opponent_vector = stack_state_to_vector(opponent.stack_state)
        input_vector = np.concatenate(
            [player_vector, opponent_vector] + [value_to_one_hot(card.value) for card in player.hand] +
            [np.zeros(60)] * (6 - len(player.hand))
        )
        return input_vector


    def evaluate_state(self, opponent):
        """ evaluate player state score """
        # we search to minize the increase of our increasing stack
        self_score = 60.0 + (self.decreasing_list[-1].value - self.increasing_list[-1].value) 
        self_factor = 5.0
        # we look for maximizing our opponent stack difference
        opp_score  = 60.0 + (opponent.increasing_list[-1].value - opponent.decreasing_list[-1].value) 
        opp_factor = 1.0
        return self_factor * self_score + opp_factor * opp_score

    def get_random_action(self, opponent):
        """ determine a playable action randomly
            returns None if no action can be played """
        possible_actions = sum([self.register_valid_actions(opponent, card) for card in self.hand], [])
        if len(possible_actions) != 0:
            player_action = random.choice(possible_actions)
            player_action.cost = -17
            return player_action
        return None

    def random_play(self, game):
        """ play randomly from the set of valid actions
            return True if a valid play was made, False otherwise """
        possible_actions = sum([self.register_valid_actions(game.get_opponent(0), card) for card in self.hand], [])
        opponent = game.get_opponent(0)
        player_action = self.get_random_action(opponent)
        if player_action is None:
            return False
        else:
            self.execute(player_action, opponent)
            True

    def reset_turn(self):
        """ reset player turn state """
        super().reset_turn()
        self.available_action = None

    def educated_play(self, opponent):
        """ use NN to determine the next action to play """
        # NN has been trained to guess the next two actions (one turn)
        # so we must switch between a state where we guess 2 actions, store one
        # and return the other, and a state where we returned the stored action
        if self.available_action:
            action = self.available_action
            self.available_action = None
        else:
            action, self.available_action = self.get_action_pairs(opponent)
        return action

    def get_action_pairs(self, opponent):
        """ use NN to build the next action pair """
        current_state = self.get_state(opponent)
        target_vec = self.model.predict(np.array([current_state]))[0]
        a = np.argmax(target_vec)
        a0 = int(a % 24)
        a1 = int(a / 24)
        card0_id = int(a0 % 6)
        action0_id = int(a0 / 6)
        card1_id = int(a1 % 6)
        action1_id = int(a1 / 6)
        # WARNING: beware card1 is chosen after card0 has been removed from hand
        # (this should be implemented the same way during the training)
        action0 = self.get_action(card0_id, action0_id, cost=1)
        action1 = self.get_action(card1_id, action1_id, cost=2)
        return action0, action1

    def reset_statistics(self):
        """ reset play validity statistics """
        self.play_count = 0
        self.valid_count_action0 = 0
        self.valid_count_action1 = 0

    def get_action_to_play(self, opponent):
        """ determine an action to play using the training NN """
        player_action = self.educated_play(opponent)
        if player_action is None:
            verbose_report("invalid card from educated play")
            verbose_report("fallback to random play")
            player_action = self.get_random_action(opponent)
        elif not self.is_action_valid(player_action, opponent):
            verbose_report("invalid action from educated play")
            verbose_report("fallback to random play")
            player_action = self.get_random_action(opponent)
        else:
            verbose_report("valid action")
            if player_action.cost == 1:
                self.valid_count_action0 += 1
            elif player_action.cost == 2:
                self.valid_count_action1 += 1
        self.play_count += 1
        return player_action

    def get_action(self, card_id, action_id, cost=None):
        if card_id >= len(self.hand) or action_id > 4:
            return None
        else:
            return Action(self.hand[card_id], ACTION_LIST[action_id], cost=cost)

    def train_model(self, num_party=100, epochs=5):
        print("training model")

        WIN_BONUS = 5000
        LOSS_MALUS = -5000
        INVALID_MALUS = -1000

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
                print("Episode {} of {}".format(i + 1, num_party))
            r_sum = 0
            game_ended = False
            while not game_ended:
                # start a new turn of NN player
                self.reset_turn()
                current_state = self.get_state(opponent)
                target_vec = self.model.predict(np.array([current_state]))[0]
                if np.random.random() < eps:
                    # random input to implement epsilon-greedy policy
                    action0 = np.random.randint(0, 24)
                    action1 = np.random.randint(0, 24)
                    a = action0 + 24 * action1
                else:
                    a= np.argmax(target_vec)
                    action0 = int(a % 24)
                    action1 = int(a / 24)
                card0_id = int(action0 % 6)
                action0_id = int(action0 / 6)
                card1_id = int(action1 % 6)
                action1_id = int(action1 / 6)

                action0_obj = self.get_action(card0_id, action0_id)
                action1_obj = self.get_action(card1_id, action1_id)
                # initial reward
                reward = 0
                remaining_action = 0
                invalid_play = True
                if action0_obj is None or action1_obj is None:
                    # at least one invalid card
                    reward = 0
                    remaining_action = 2
                else:
                    # valid cards
                    opponent = game.get_opponent(0)
                    if not self.is_action_valid(action0_obj, opponent):
                        # at least one invalid action
                        reward = 0
                        remaining_action = 2
                    else:
                        self.execute(action0_obj, opponent)
                        if not self.is_action_valid(action1_obj, opponent):
                            reward = 0
                            remaining_action = 1
                        else:
                            reward += self.evaluate_state(opponent)
                            self.execute(action1_obj, opponent)
                            reward += self.evaluate_state(opponent)
                            remaining_action = 0
                            invalid_play = False
                game_ended = False
                while not game_ended and remaining_action > 0:
                    # random play to bridge missing actions
                    game_ended = not self.random_play(game)
                    remaining_action -= 1

                if not game_ended and self.win_condition():
                        # reward bonus
                        reward += WIN_BONUS
                        game_ended = True

                if not game_ended:
                    # plays opponent turns
                    game_ended = not game.play_one_turn(1)
                    if game_ended:
                        # game stopped, opponent has lost
                        reward += WIN_BONUS
                        game_ended = True
                    elif opponent.win_condition():
                        # check if opponent win in this stage
                        # reward malus
                        reward += LOSS_MALUS
                        game_ended = True

                next_state = self.get_state(opponent)

                if invalid_play:
                    reward = 0
                    target = reward
                elif game_ended:
                    # no next state
                    target = reward
                else:
                    # valid and next state
                    target = reward + y * np.max(self.model.predict(np.array([next_state])))
                target_vec[a] = target
                self.model.fit(np.array([current_state]), np.array([target_vec]), epochs=1, verbose=0)
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


def main_trainable_player():
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

def main_knowledgeable_player():
    parser = argparse.ArgumentParser(description="NN based IA for the Game")
    parser.add_argument("--num", type=int, default=100, help="number of party to simulate for NN training")
    parser.add_argument("--epochs", type=int, default=5, help="number of epoch to run for training")
    parser.add_argument("--save-file", type=str, default=None, help="NN weights will be saved to this file")
    parser.add_argument("--load-file", type=str, default=None, help="NN weights will be stored from this file (bypass training)")
    parser.add_argument("--num-eval-game", type=int, default=10, help="number of game(s) simulated for player evaluation")
    parser.add_argument("--swap", action="store_const", default=False, const=True, help="swap NN and random player during eval games")
    parser.add_argument("--skip-training", action="store_const", default=False, const=True, help="skip training phase")
    args = parser.parse_args()

    # train and evaluate model
    nn_player = KnowledgablePlayer()
    nn_player.build_model()
    if args.load_file:
        print("loading model from {}".format(args.load_file))
        nn_player.model.load_weights(args.load_file)

    if not args.skip_training:
        nn_player.train_model(args.num, args.epochs)
    if args.save_file:
        print("saving model in {}".format(args.save_file))
        nn_player.model.save_weights(args.save_file)

    print("evaluating NN during {} game(s)".format(args.num_eval_game))
    # execute model on one game
    if args.swap:
        game = Game(RandomPlayer(), nn_player)
    else:
        game = Game(nn_player, RandomPlayer())
    win_counts = [0, 0]
    nn_player.reset_statistics()

    for i in range(args.num_eval_game):
        # simulating a new game
        game.reset()
        player_id = 0
        winner_id = game.play_game()
        win_counts[winner_id] += 1
    for winner_id in range(2):
        print("player {} ({}) wins {} / {} = {:.2f} % !".format(winner_id, game.players[winner_id].__class__.__name__, win_counts[winner_id], args.num_eval_game, win_counts[winner_id] / args.num_eval_game * 100))

    valid_count = nn_player.valid_count_action0 + nn_player.valid_count_action1
    print("NN player played {}({} + {})/{} valid actions {:.2f}".format(
        valid_count,
        nn_player.valid_count_action0,
        nn_player.valid_count_action1,
        nn_player.play_count,
        100 * valid_count / nn_player.play_count))


if __name__ == "__main__":
    main_knowledgeable_player()

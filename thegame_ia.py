# -*- coding: utf-8 -*-
import random
import argparse

from basic_player import BasePlayer
from utils import verbose_report
import utils


class RandomPlayer(BasePlayer):
    """ player whose play is any valid card """

    def get_action_to_play(self, opponent):
        playable_cards = self.get_playable_cards(opponent)
        if not len(playable_cards):
            # no card could be played
            return None
        card = random.choice(playable_cards)
        possible_actions = self.register_valid_actions(opponent, card)
        action = random.choice(possible_actions)
        return action

    def cost_play_on_increasing(self, opp, card):
        return 0
    def cost_play_on_decreasing(self, opp, card):
        return 0
    def cost_play_on_opp_increasing(self, opp, card):
        return 0
    def cost_play_on_opp_decreasing(self, opp, card):
        return 0

class StarterPlayer(BasePlayer):
    def get_action_to_play(self, opponent):
        playable_cards = self.get_playable_cards(opponent)
        actions = sum([self.register_valid_actions(opponent, card) for card in playable_cards], [])
        if len(actions) == 0:
            return None
        action = min(actions, key=lambda action: action.cost)
        return action

    def cost_play_on_increasing(self, opp, card):
        return abs(self.increasing_list[-1].value - card.value)
    def cost_play_on_decreasing(self, opp, card):
        return abs(self.decreasing_list[-1].value - card.value)
    def cost_play_on_opp_increasing(self, opp, card):
        return 120 - abs(opp.increasing_list[-1].value - card.value)
    def cost_play_on_opp_decreasing(self, opp, card):
        return 120 - abs(opp.decreasing_list[-1].value - card.value)

class MediumPlayer(StarterPlayer):
    # cost of 0 allows to only plays +/- 10 diff cards on self stacks
    MAXIMAL_EXTRA_PLAY_THRESHOLD = 0

    def cost_play_on_increasing(self, opp, card):
        if False and card.value == self.increasing_list[-1].value - 10:
            # interesting play
            return 0
        else:
            return abs(self.increasing_list[-1].value - card.value)
    def cost_play_on_decreasing(self, opp, card):
        if False and card.value == self.decreasing_list[-1].value + 10:
            # interesting play
            return 0
        else:
            return abs(self.decreasing_list[-1].value - card.value)
    def cost_play_on_opp_increasing(self, opp, card):
        return 120 + abs(opp.increasing_list[-1].value - card.value)
    def cost_play_on_opp_decreasing(self, opp, card):
        return 120 + abs(opp.decreasing_list[-1].value - card.value)

    def get_extra_card_to_play(self, opponent):
        playable_cards = self.get_playable_cards(opponent)
        # if no card is playable, we wait for next turn
        if len(playable_cards) == 0:
            return None

        actions = sum([self.register_valid_actions(opponent, card) for card in playable_cards], [])
        action = min(actions, key=lambda action: action.cost)
        if action.cost > MediumPlayer.MAXIMAL_EXTRA_PLAY_THRESHOLD:
            # if the extra play is not interesting we do not play it
            return None
        return action

class TrainablePlayer(StarterPlayer):
    def play_state(self, opponent):
        state = [
            self.increasing_list[-1].value,
            self.decreasing_list[-1].value,
            opponent.increasing_list[-1].value,
            opponent.decreasing_list[-1].value,
        ]
        # adding hand state
        state += [card.value for card in self.hand]

PLAYER_STATE_TEMPLATE = """player {}'s
    hand is {}
    increasing is {}
    decreasing is {}"""

class Game:
    def __init__(self, player0, player1):
        #self.players = [MediumPlayer(), StarterPlayer()]
        self.players = [player0, player1]

    def reset(self):
        for player in self.players:
            player.reset()
        # initial draw (only 4, 2 more card will be picked up at beginning of first turn)
        self.players[0].pick_card_from_stack(4)
        self.players[1].pick_card_from_stack(4)

    def play_one_turn(self, player_id=0):
        verbose_report("player {} turn".format(player_id))
        if not self.players[player_id].play_one_turn(self.players[1 - player_id]):
            verbose_report("player {} lost !".format(player_id))
            return False
        return True

    def play_game(self):
        player_id = 0
        while self.play_one_turn(player_id):
            if self.players[player_id].win_condition():
                verbose_report("player {} has won".format(player_id))
                return player_id
            player_id = 1 - player_id

        #self.players[player_id].display_state(str(player_id))
        #self.players[1 - player_id].display_state(str(1 - player_id))
        return 1 - player_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='IA design and evaluation framework for The Game')
    parser.add_argument('--num', type=int, help='number of party to simulate')
    parser.add_argument('--verbose', action="store_const", const=True, default=False, help='activate verbose message')
    args = parser.parse_args()

    NUM_GAMES = args.num
    utils.VERBOSE_ENABLED = args.verbose

    #PLAYER_CLASS = [RandomPlayer, StarterPlayer, MediumPlayer]
    PLAYER_CLASS = [StarterPlayer, MediumPlayer]
    for Player0Class in PLAYER_CLASS:
        for Player1Class in PLAYER_CLASS:
            win_count = [0, 0]
            game = Game(Player0Class(), Player1Class())
            for i in range(NUM_GAMES):
                game.reset()
                winner_id = game.play_game()
                win_count[winner_id] += 1
            print(Player0Class.__name__, Player1Class.__name__, ["{:.2f}".format(p / NUM_GAMES * 100) for p in win_count])

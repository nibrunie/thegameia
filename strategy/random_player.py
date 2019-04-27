# -*- coding: utf-8 -*-
import random

from basic_player import BasePlayer, thegameia_strategy

@thegameia_strategy
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

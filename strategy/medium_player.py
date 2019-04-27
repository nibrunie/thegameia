# -*- coding: utf-8 -*-
import random

from basic_player import BasePlayer, thegameia_strategy

@thegameia_strategy
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

@thegameia_strategy
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


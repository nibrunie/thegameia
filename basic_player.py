# -*- coding: utf-8 -*-

import random

from cards import SelfCard, OppCard
from utils import verbose_report

class ForbiddenPlay(Exception):
    """ error raised whan an unauthorized play is intended """
    pass


class BasePlayer:
    def __init__(self):
        self.reset()

    def reset(self):
        self.full_list = [SelfCard(i) for i in (range(2, 60))]
        random.shuffle(self.full_list)
        self.increasing_list = [SelfCard(1)]
        self.decreasing_list = [SelfCard(60)]
        self.hand = []
        # does the player has already played a card on one of his opponent
        # stacks this turn
        self.has_played_on_opp_this_turn = False

    def reset_turn(self):
        if self.has_played_on_opp_this_turn:
            # refill hand completely
            self.pick_card_from_stack(n=6 - len(self.hand))
        else:
            self.pick_card_from_stack(n=2)
        self.has_played_on_opp_this_turn = False

    def end_turn(self):
        pass

    def pick_card_from_stack(self, n=1):
        self.hand += [self.full_list.pop(0) for i in range(min(n, len(self.full_list)))]

    def play_on_increasing(self, card):
        verbose_report("plays {} on increasing".format(card))
        if self.is_valid_play_on_increasing(card):
            self.use_hand_card(card)
            self.increasing_list.append(card)
        else:
            raise ForbiddenPlay

    def play_on_opponent_increasing(self, opponent, card):
        verbose_report("plays {} on opponent increasing".format(card))
        if self.is_valid_play_on_opponent_increasing(opponent, card):
            self.use_hand_card(card)
            self.has_played_on_opp_this_turn = True
            opponent.increasing_list.append(card.get_opp())
        else:
            raise ForbiddenPlay

    def play_on_decreasing(self, card):
        verbose_report("plays {} on decreasing".format(card))
        if self.is_valid_play_on_decreasing(card):
            self.use_hand_card(card)
            self.decreasing_list.append(card)
        else:
            raise ForbiddenPlay

    def play_on_opponent_decreasing(self, opponent, card):
        verbose_report("plays {} on opponent decreasing".format(card))
        if self.is_valid_play_on_opponent_decreasing(opponent, card):
            self.use_hand_card(card)
            opponent.decreasing_list.append(card.get_opp())
            self.has_played_on_opp_this_turn = True
        else:
            raise ForbiddenPlay

    def is_valid_play_on_opponent_increasing(self, opponent, card):
        return not self.has_played_on_opp_this_turn and card < opponent.increasing_list[-1]

    def is_valid_play_on_opponent_decreasing(self, opponent, card):
        return not self.has_played_on_opp_this_turn and card > opponent.decreasing_list[-1]

    def is_valid_play_on_increasing(self, card):
        return card > self.increasing_list[-1] or card == self.increasing_list[-1] - 10

    def is_valid_play_on_decreasing(self, card):
        return card < self.decreasing_list[-1] or card == self.decreasing_list[-1] + 10

    def cost_play_on_increasing(self, opp, card):
        raise NotImplementedError
    def cost_play_on_decreasing(self, opp, card):
        raise NotImplementedError
    def cost_play_on_opp_increasing(self, opp, card):
        raise NotImplementedError
    def cost_play_on_opp_decreasing(self, opp, card):
        raise NotImplementedError

    def register_valid_plays(self, opponent, card):
        plays = []
        if self.is_valid_play_on_increasing(card):
            plays.append((
                self.cost_play_on_increasing(opponent, card),
                lambda: self.play_on_increasing(card)))
        if self.is_valid_play_on_decreasing(card):
            plays.append((
                self.cost_play_on_decreasing(opponent, card),
                lambda: self.play_on_decreasing(card)
                ))
        if self.is_valid_play_on_opponent_increasing(opponent, card):
            plays.append((
                self.cost_play_on_opp_increasing(opponent, card),
                lambda: self.play_on_opponent_increasing(opponent, card)))
        if self.is_valid_play_on_opponent_decreasing(opponent, card):
            plays.append((
                self.cost_play_on_opp_decreasing(opponent, card),
                lambda: self.play_on_opponent_decreasing(opponent, card)))
        return plays

    def has_valid_play(self, opponent, card):
        return self.is_valid_play_on_increasing(card) or \
            self.is_valid_play_on_decreasing(card) or \
            self.is_valid_play_on_opponent_increasing(opponent, card) or \
            self.is_valid_play_on_opponent_decreasing(opponent, card)

    def use_hand_card(self, card):
        self.hand.remove(card)
        return card

    def get_card_to_play(self, opponent):
        """ function returning a tuple (card, play) indicating which
            card and which action the player wants to execute next """
        raise NotImplementedError
    def get_extra_card_to_play(self, opponent):
        """ once the player has placed two cards, he can (if he wishes)
            try to play other card

            returning None, None indicates that the play does not want to
            play extra cards """
        return None, None

    def display_state(self, name=""):
        print(PLAYER_STATE_TEMPLATE.format(name, self.hand, self.increasing_list, self.decreasing_list))

    def win_condition(self):
        return len(self.hand) == 0 and len(self.full_list) == 0

    def get_playable_cards(self, opponent):
        playable_cards = [card for card in self.hand if self.has_valid_play(opponent, card)]
        return playable_cards

    def play_one_turn(self, opponent):
        # reset turn initial state
        self.reset_turn()
        # two cards to be played
        for card_played in range(2):
            card, play = self.get_card_to_play(opponent)
            if card is None or play is None:
                return False
            play()
        while True:
            card, play = self.get_extra_card_to_play(opponent)
            if card is None or play is None:
                break
            play()
        # draw 2 new cards at the end of turn
        return True


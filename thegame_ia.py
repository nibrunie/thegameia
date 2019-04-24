# -*- coding: utf-8 -*-

import random

class ForbiddenPlay(Exception):
    pass

class Card:
    def __init__(self, value):
        self.value = value

    def __eq__(self, other_card):
        return self.value == other_card.value

    def __ne__(self, other_card):
        return self.value != other_card.value

    def __lt__(self, other_card):
        return self.value < other_card.value

    def __le__(self, other_card):
        return self.value <= other_card.value

    def __gt__(self, other_card):
        return self.value > other_card.value

    def __ge__(self, other_card):
        return self.value >= other_card.value

    def offset(self, diff=0):
        return self.__class__(self.value + diff)

    def __sub__(self, diff):
        return self.offset(-diff)
    def __add__(self, diff):
        return self.offset(diff)

class OppCard(Card):
    def __str__(self):
        return "O({})".format(self.value)
    def __repr__(self):
        return "O({})".format(self.value)

class SelfCard(Card):
    def __str__(self):
        return "S({})".format(self.value)
    def __repr__(self):
        return "S({})".format(self.value)

    def get_opp(self):
        return OppCard(self.value)


class Player:
    def __init__(self):
        self.full_list = [SelfCard(i) for i in (range(2, 60))]
        random.shuffle(self.full_list)
        self.increasing_list = [SelfCard(1)]
        self.decreasing_list = [SelfCard(60)]
        self.hand = []

    def pick_card_from_stack(self, n=1):
        self.hand += [self.full_list.pop(0) for i in range(min(n, len(self.full_list)))]

    def play_on_increasing(self, card):
        print("plays {} on increasing".format(card))
        if self.is_valid_play_on_increasing(card):
            self.increasing_list.append(card)
        else:
            raise ForbiddenPlay

    def play_on_opponent_increasing(self, opponent, card):
        print("plays {} on opponent increasing".format(card))
        if self.is_valid_play_on_opponent_increasing(opponent, card):
            opponent.increasing_list.append(card.get_opp())
        else:
            raise ForbiddenPlay

    def play_on_decreasing(self, card):
        print("plays {} on decreasing".format(card))
        if self.is_valid_play_on_decreasing(card):
            self.decreasing_list.append(card)
        else:
            raise ForbiddenPlay

    def play_on_opponent_decreasing(self, opponent, card):
        print("plays {} on opponent decreasing".format(card))
        if self.is_valid_play_on_opponent_decreasing(opponent, card):
            opponent.decreasing_list.append(card.get_opp())
        else:
            raise ForbiddenPlay

    def is_valid_play_on_opponent_increasing(self, opponent, card):
        return card < opponent.increasing_list[-1]

    def is_valid_play_on_opponent_decreasing(self, opponent, card):
        return card > opponent.increasing_list[-1]

    def is_valid_play_on_increasing(self, card):
        return card > self.increasing_list[-1] or card == self.increasing_list[-1] - 10

    def is_valid_play_on_decreasing(self, card):
        return card < self.decreasing_list[-1] or card == self.decreasing_list[-1] + 10

    def register_valid_plays(self, opponent, card):
        plays = []
        if self.is_valid_play_on_increasing(card):
            plays.append(lambda v: self.play_on_increasing(v))
        if self.is_valid_play_on_decreasing(card):
            plays.append(lambda v: self.play_on_decreasing(v))
        if self.is_valid_play_on_opponent_increasing(opponent, card):
            plays.append(lambda v: self.play_on_opponent_increasing(opponent, v))
        if self.is_valid_play_on_opponent_decreasing(opponent, card):
            plays.append(lambda v: self.play_on_opponent_decreasing(opponent, v))
        return plays

    def has_valid_play(self, opponent, card):
        return self.is_valid_play_on_increasing(card) or \
            self.is_valid_play_on_decreasing(card) or \
            self.is_valid_play_on_opponent_increasing(opponent, card) or \
            self.is_valid_play_on_opponent_decreasing(opponent, card)

    def use_hand_card(self, card):
        self.hand.remove(card)
        return card

    def play_one_turn(self, opponent):
        # two cards to be played
        for card_played in range(2):
            playable_cards = [card for card in self.hand if self.has_valid_play(opponent, card)]
            if not len(playable_cards):
                # no card could be played
                return False
            card = self.use_hand_card(random.choice(playable_cards))
            plays = self.register_valid_plays(opponent, card)
            play = random.choice(plays)
            play(card)
        # draw 2 new cards at the end of turn
        self.pick_card_from_stack(n=2)
        return True

    def display_state(self, name=""):
        print(PLAYER_STATE_TEMPLATE.format(name, self.hand, self.increasing_list, self.decreasing_list))


PLAYER_STATE_TEMPLATE = """player {}'s
    hand is {}
    increasing is {}
    decreasing is {}"""

class Game:
    def __init__(self):
        self.players = [Player(), Player()]
        # initial draw
        self.players[0].pick_card_from_stack(6)
        self.players[1].pick_card_from_stack(6)

    def play_one_turn(self, player_id=0):
        print("player {} turn".format(player_id))
        if not self.players[player_id].play_one_turn(self.players[1 - player_id]):
            print("player {} lost !".format(player_id))
            return False
        return True

    def play_game(self):
        player_id = 0
        while self.play_one_turn(player_id):
            player_id = 1 - player_id

        self.players[player_id].display_state(str(player_id))
        self.players[1 - player_id].display_state(str(1 - player_id))


if __name__ == "__main__":
    game = Game()
    game.play_game()

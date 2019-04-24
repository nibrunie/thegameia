# -*- coding: utf-8 -*-

import random

class ForbiddenPlay(Exception):
    pass


class Player:
    def __init__(self):
        self.full_list = list(range(2, 60))
        random.shuffle(self.full_list)
        self.increasing_list = [1]
        self.decreasing_list = [60]
        self.hand = []

    def pick_card_from_stack(self, n=1):
        self.hand += [self.full_list.pop(0) for i in range(min(n, len(self.full_list)))]

    def play_on_increasing(self, value):
        print("plays {} on increasing".format(value))
        if self.is_valid_play_on_increasing(value):
            self.increasing_list.append(value)
        else:
            raise ForbiddenPlay

    def play_on_opponent_increasing(self, opponent, value):
        print("plays {} on opponent increasing".format(value))
        if self.is_valid_play_on_opponent_increasing(opponent, value):
            opponent.increasing_list.append(value)
        else:
            raise ForbiddenPlay

    def play_on_decreasing(self, value):
        print("plays {} on decreasing".format(value))
        if self.is_valid_play_on_decreasing(value):
            self.decreasing_list.append(value)
        else:
            raise ForbiddenPlay

    def play_on_opponent_decreasing(self, opponent, value):
        print("plays {} on opponent decreasing".format(value))
        if self.is_valid_play_on_opponent_decreasing(opponent, value):
            opponent.decreasing_list.append(value)
        else:
            raise ForbiddenPlay

    def is_valid_play_on_opponent_increasing(self, opponent, value):
        return value < opponent.increasing_list[-1]

    def is_valid_play_on_opponent_decreasing(self, opponent, value):
        return value > opponent.increasing_list[-1]

    def is_valid_play_on_increasing(self, value):
        return value > self.increasing_list[-1] or value == self.increasing_list[-1] - 10

    def is_valid_play_on_decreasing(self, value):
        return value < self.decreasing_list[-1] or value == self.decreasing_list[-1] + 10

    def register_valid_plays(self, opponent, value):
        plays = []
        if self.is_valid_play_on_increasing(value):
            plays.append(lambda v: self.play_on_increasing(v))
        if self.is_valid_play_on_decreasing(value):
            plays.append(lambda v: self.play_on_decreasing(v))
        if self.is_valid_play_on_opponent_increasing(opponent, value):
            plays.append(lambda v: self.play_on_opponent_increasing(opponent, v))
        if self.is_valid_play_on_opponent_decreasing(opponent, value):
            plays.append(lambda v: self.play_on_opponent_decreasing(opponent, v))
        return plays

    def has_valid_play(self, opponent, value):
        return self.is_valid_play_on_increasing(value) or \
            self.is_valid_play_on_decreasing(value) or \
            self.is_valid_play_on_opponent_increasing(opponent, value) or \
            self.is_valid_play_on_opponent_decreasing(opponent, value)

    def use_hand_card(self, value):
        self.hand.remove(value)
        return value

    def play_one_turn(self, opponent):
        # two cards to be played
        for card_played in range(2):
            playable_cards = [value for value in self.hand if self.has_valid_play(opponent, value)]
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


if __name__ == "__main__":
    game = Game()
    game.play_game()

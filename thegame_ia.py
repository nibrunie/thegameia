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


class BasePlayer:
    def __init__(self):
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
        print("plays {} on increasing".format(card))
        if self.is_valid_play_on_increasing(card):
            self.use_hand_card(card)
            self.increasing_list.append(card)
        else:
            raise ForbiddenPlay

    def play_on_opponent_increasing(self, opponent, card):
        print("plays {} on opponent increasing".format(card))
        if self.is_valid_play_on_opponent_increasing(opponent, card):
            self.use_hand_card(card)
            self.has_played_on_opp_this_turn = True
            opponent.increasing_list.append(card.get_opp())
        else:
            raise ForbiddenPlay

    def play_on_decreasing(self, card):
        print("plays {} on decreasing".format(card))
        if self.is_valid_play_on_decreasing(card):
            self.use_hand_card(card)
            self.decreasing_list.append(card)
        else:
            raise ForbiddenPlay

    def play_on_opponent_decreasing(self, opponent, card):
        print("plays {} on opponent decreasing".format(card))
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

    def play_one_turn(self, opponent):
        raise NotImplementedError
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
        # draw 2 new cards at the end of turn
        return True

class RandomPlayer(BasePlayer):
    """ player whose play is any valid card """

    def get_card_to_play(self, opponent):
        playable_cards = self.get_playable_cards(opponent)
        if not len(playable_cards):
            # no card could be played
            return None, None
        card = random.choice(playable_cards)
        plays = self.register_valid_plays(opponent, card)
        cost, play = random.choice(plays)
        return card, play

    def cost_play_on_increasing(self, opp, card):
        return 0
    def cost_play_on_decreasing(self, opp, card):
        return 0
    def cost_play_on_opp_increasing(self, opp, card):
        return 0
    def cost_play_on_opp_decreasing(self, opp, card):
        return 0

class StarterPlayer(BasePlayer):
    def get_card_to_play(self, opponent):
        playable_cards = self.get_playable_cards(opponent)
        plays = sum([[(card, cost_play) for cost_play in self.register_valid_plays(opponent, card)] for card in playable_cards], [])
        card, (cost, play) = min(plays, key=lambda triplet: triplet[1][0])
        return card, play

    def cost_play_on_increasing(self, opp, card):
        return abs(self.increasing_list[-1].value - card.value)
    def cost_play_on_decreasing(self, opp, card):
        return abs(self.decreasing_list[-1].value - card.value)
    def cost_play_on_opp_increasing(self, opp, card):
        return 120 - abs(opp.increasing_list[-1].value - card.value)
    def cost_play_on_opp_decreasing(self, opp, card):
        return 120 - abs(opp.decreasing_list[-1].value - card.value)


PLAYER_STATE_TEMPLATE = """player {}'s
    hand is {}
    increasing is {}
    decreasing is {}"""

class Game:
    def __init__(self):
        self.players = [RandomPlayer(), StarterPlayer()]
        # initial draw (only 4, 2 more card will be picked up at beginning of first turn)
        self.players[0].pick_card_from_stack(4)
        self.players[1].pick_card_from_stack(4)

    def play_one_turn(self, player_id=0):
        print("player {} turn".format(player_id))
        if not self.players[player_id].play_one_turn(self.players[1 - player_id]):
            print("player {} lost !".format(player_id))
            return False
        return True

    def play_game(self):
        player_id = 0
        while self.play_one_turn(player_id):
            if self.players[player_id].win_condition():
                print("player {} has won".format(player_id))
                break
            player_id = 1 - player_id

        self.players[player_id].display_state(str(player_id))
        self.players[1 - player_id].display_state(str(1 - player_id))


if __name__ == "__main__":
    game = Game()
    game.play_game()

# -*- coding: utf-8 -*-

class Card:
    """ generic game card """
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
    """ card played by the opponent player on the other player stacks """
    def __str__(self):
        return "O({})".format(self.value)
    def __repr__(self):
        return "O({})".format(self.value)

class SelfCard(Card):
    """ card played by one player on its own stacks """
    def __str__(self):
        return "S({})".format(self.value)
    def __repr__(self):
        return "S({})".format(self.value)

    def get_opp(self):
        return OppCard(self.value)


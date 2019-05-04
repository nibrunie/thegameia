# -*- coding: utf-8 -*-

import numpy as np


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

# -*- coding: utf-8 -*-
import random
import argparse

import utils
from strategy import *

from game_simulator import Game
from basic_player import StrategyRegister



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='IA design and evaluation framework for The Game')
    parser.add_argument('--num', type=int, help='number of party to simulate')
    parser.add_argument('--verbose', action="store_const", const=True, default=False, help='activate verbose message')
    parser.add_argument('--strategy-list', default="StarterPlayer,MediumPlayer", type=str, help='list of strategies to be tested among {}'.format(", ".join(k for k in StrategyRegister.strategy_map)))
    args = parser.parse_args()

    NUM_GAMES = args.num
    utils.VERBOSE_ENABLED = args.verbose

    try:
        PLAYER_CLASS = [StrategyRegister.strategy_map[strategy] for strategy in args.strategy_list.split(',')]
    except KeyError as e:
        print(StrategyRegister.strategy_map)
        raise e
    for Player0Class in PLAYER_CLASS:
        for Player1Class in PLAYER_CLASS:
            win_count = [0, 0]
            game = Game(Player0Class(), Player1Class())
            for i in range(NUM_GAMES):
                game.reset()
                winner_id = game.play_game()
                win_count[winner_id] += 1
            print(Player0Class.__name__, Player1Class.__name__, ["{:.2f}".format(p / NUM_GAMES * 100) for p in win_count])

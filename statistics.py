# -*- coding: utf-8 -*-
import random

import utils

from game_simulator import Game

def evaluate_strategy(strategy, opponent_list, num_eval_game=10):
    """ Benchmark player object @strategy to evaluate its probability to win """
    win_counts = 0
    strategy.reset_statistics()

    for i in range(num_eval_game):
        opponent = random.choice(opponent_list)
        strategy_id = random.choice([0, 1])
        if strategy_id == 1:
            game = Game(opponent, strategy)
        else:
            assert strategy_id == 0
            game = Game(strategy, opponent)
        # simulating a new game
        game.reset()
        winner_id = game.play_game()
        if winner_id == strategy_id:
            win_counts += 1
    proba_win = win_counts / num_eval_game
    print("strategy player ({}) wins {} / {} = {:.2f} % !".format(
        strategy.__class__.__name__,
        win_counts,
        num_eval_game,
        proba_win * 100))

    strategy.display_statistics()
    return proba_win

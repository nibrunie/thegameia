# -*- coding: utf-8 -*-

from utils import verbose_report


class Game:
    """ Game emaluation object """
    def __init__(self, player0, player1):
        #self.players = [MediumPlayer(), StarterPlayer()]
        self.players = [player0, player1]

    def reset(self):
        for player in self.players:
            player.reset()
        # initial draw (only 4, 2 more card will be picked up at beginning of first turn)
        self.players[0].pick_card_from_stack(4)
        self.players[1].pick_card_from_stack(4)

    def play_one_turn(self, player_id=0):
        verbose_report("player {} turn".format(player_id))
        if not self.players[player_id].play_one_turn(self.get_opponent(player_id)):
            verbose_report("player {} lost !".format(player_id))
            return False
        return True

    def get_player_state(self, player_id):
        return self.players[player_id].state

    def get_opponent(self, player_id):
        return self.players[1 - player_id]

    def play_game(self):
        player_id = 0
        while self.play_one_turn(player_id):
            if self.players[player_id].win_condition():
                verbose_report("player {} has won".format(player_id))
                return player_id
            player_id = 1 - player_id

        #self.players[player_id].display_state(str(player_id))
        #self.players[1 - player_id].display_state(str(1 - player_id))
        return 1 - player_id



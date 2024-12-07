import numpy as np
from game_2048.tiles import Tiles, Tile

class Game2048Env:
    def __init__(self, game_instance):
        self.game = game_instance

    def reset(self):
        self.game.reset()
        return self.get_state()

    def get_state(self):
        return np.array(self.game.get_board()).flatten()

    def step(self, action):
        moved, reward, done = self.game.move(action)
        # Ensure `moved` is a Boolean and log to confirm its type
        moved = bool(moved)
        #print(f"[DEBUG] Game2048Env.step() -> Action: {action}, Moved (Boolean): {moved}, Reward: {reward}, Done: {done}")
        return moved, reward, done  # Explicitly return `moved` as Boolean

    def get_score(self):
        return self.game.get_score()

    def clone(self):
        return Game2048Env(self.game.clone())

    def check_game_over(self):
        return self.game.check_game_over()

    def get_empty_cells(self):
        return self.game.get_empty_cells()

    def spawn_tile(self, x, y, value):
        self.game.tiles.append(Tile(x, y, value))

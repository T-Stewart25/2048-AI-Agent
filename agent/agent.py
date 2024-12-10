import numpy as np

class ExpectimaxAgent:
    def __init__(self, env, max_depth=3):
        self.env = env
        self.max_depth = max_depth
        self.action_map = {0: 'up', 1: 'down', 2: 'left', 3: 'right'}

    def get_best_move(self, state):
        best_score = -float('inf')
        best_move = None

        for move_int, move_str in self.action_map.items():
            new_env = self.env.clone()
            moved, _, _ = new_env.step(move_str)
            if not isinstance(moved, bool):
                moved = bool(moved)
            if moved:
                score = self.expectimax(new_env, depth=1, is_maximizing=False)
                if score > best_score:
                    best_score = score
                    best_move = move_str

        return best_move

    def expectimax(self, env, depth, is_maximizing):
        if depth == self.max_depth or env.check_game_over():
            return self.evaluate(env)

        if is_maximizing:
            max_score = -float('inf')
            for move_str in self.action_map.values():
                new_env = env.clone()
                moved, _, _ = new_env.step(move_str)
                if not isinstance(moved, bool):
                    moved = bool(moved)
                if moved:
                    score = self.expectimax(new_env, depth + 1, is_maximizing=False)
                    max_score = max(max_score, score)
            return max_score
        else:
            scores = []
            empty_cells = env.get_empty_cells()
            if not empty_cells:
                return self.evaluate(env)
            for (x, y) in empty_cells:
                for tile_value, probability in [(2, 0.9), (4, 0.1)]:
                    new_env = env.clone()
                    new_env.spawn_tile(x, y, tile_value)
                    score = self.expectimax(new_env, depth + 1, is_maximizing=True)
                    scores.append(score * probability)
            return sum(scores)/len(scores) if scores else 0

    def evaluate(self, env):
        score = env.get_score()
        empty_cells = len(env.get_empty_cells())
        return score + empty_cells * 100

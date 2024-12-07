import numpy as np
from game_2048.game_environment import Game2048Env
from game_2048.tiles import Tiles

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
            moved, score, _ = new_env.step(move_str)

            # Debug to confirm the type and value of `moved`
            #print(f"[DEBUG] get_best_move -> Move: {move_str}, Moved (type: {type(moved)}): {moved}, Score: {score}")

            if not isinstance(moved, bool):
                print("[ERROR] `moved` is not a Boolean in get_best_move, converting explicitly.")
                moved = bool(moved)  # Convert to Boolean

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
                moved, score, _ = new_env.step(move_str)


                if not isinstance(moved, bool):
                    print("[ERROR] `moved` is not a Boolean in expectimax, converting explicitly.")
                    moved = bool(moved)

                if moved:
                    score = self.expectimax(new_env, depth + 1, is_maximizing=False)
                    max_score = max(max_score, score)
            return max_score
        else:
            scores = []
            empty_cells = env.get_empty_cells()
            for (x, y) in empty_cells:
                for tile_value, probability in [(2, 0.9), (4, 0.1)]:
                    new_env = env.clone()
                    new_env.spawn_tile(x, y, tile_value)
                    score = self.expectimax(new_env, depth + 1, is_maximizing=True)
                    scores.append(score * probability)

            return sum(scores) / len(scores) if scores else 0

    def evaluate(self, env):
        score = env.get_score()
        empty_cells = len(env.get_empty_cells())
        return score + empty_cells * 100

def run_expectimax_agent(episodes, results_queue, index):
    game = Tiles()
    env = Game2048Env(game)
    agent = ExpectimaxAgent(env, max_depth=3)
    
    best_result = 0
    best_action_sequence = []

    for episode in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0
        current_action_sequence = []

        step = 0
        while not done:
            best_move = agent.get_best_move(state)
            if best_move is None:
                break

            next_state, reward, done = env.step(best_move)
            current_action_sequence.append(best_move)
            state = next_state
            total_reward += reward
            step += 1

            results_queue.put(("progress", index, episode, step, total_reward))

        if total_reward > best_result:
            best_result = total_reward
            best_action_sequence = current_action_sequence

    results_queue.put(("result", best_result, best_action_sequence))

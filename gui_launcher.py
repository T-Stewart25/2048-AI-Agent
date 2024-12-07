# gui_launcher.py
import tkinter as tk
from multiprocessing import Process, Queue
import pickle
import os
import numpy as np
import pygame

from game_2048.game_environment import Game2048Env
from game_2048.tiles import Tiles
from game_2048.setup import initialize_game, draw_grid
from agent.agent import ExpectimaxAgent
from agent.dqn_agent import DQNAgent, ACTIONS
from game_2048.play_game import run_2048_game  # Manual play function

results_queue = Queue()
all_results = {}
progress_window = None
instance_progress = {}

def update_progress():
    global progress_window, all_results
    while not results_queue.empty():
        message_type, *data = results_queue.get()
        if message_type == "progress":
            instance_index, episode, score = data
            if instance_index not in all_results:
                all_results[instance_index] = {}
            all_results[instance_index][episode] = score
            instance_progress[instance_index].config(
                text=f"Instance {instance_index+1} - Episode: {episode+1}, Score: {score}"
            )
        elif message_type == "done":
            instance_index = data[0]
            instance_progress["final"].config(
                text=f"Training for Instance {instance_index+1} complete. Waiting for others..."
            )

    num_instances = int(expectimax_entry.get())
    num_episodes = int(episode_entry.get())
    if len(all_results) == num_instances and all(
        len(all_results[i]) == num_episodes for i in all_results
    ):
        if progress_window:
            progress_window.after(1000, progress_window.destroy)
            summary_label.config(text="Training done. Press 'Aggregate Results' to see the best model.")
    else:
        if progress_window:
            progress_window.after(1000, update_progress)

def open_progress_window(num_instances):
    global progress_window
    progress_window = tk.Toplevel(root)
    progress_window.title("Training Progress")

    for i in range(num_instances):
        instance_progress[i] = tk.Label(progress_window, text=f"Instance {i+1} - Waiting for updates...")
        instance_progress[i].pack()

    instance_progress["final"] = tk.Label(progress_window, text="Training in progress...")
    instance_progress["final"].pack()

    if progress_window:
        progress_window.after(1000, update_progress)

def run_training_instance(instance_index, num_episodes, results_queue, method):
    game = Tiles()
    env = Game2048Env(game)

    if method == "expectimax":
        agent = ExpectimaxAgent(env, max_depth=3)
        for episode in range(num_episodes):
            state = env.reset()
            done = False
            total_score = 0
            while not done:
                best_move = agent.get_best_move(state)
                if best_move is None:
                    break
                moved, reward, done = env.step(best_move)
                total_score += reward
                state = env.get_state().reshape(4,4)
            results_queue.put(("progress", instance_index, episode, total_score))
        results_queue.put(("done", instance_index))

    elif method == "dqn":
        agent = DQNAgent()
        best_score = 0
        for episode in range(num_episodes):
            env.reset()
            state = env.get_state().reshape(4,4)
            episode_reward = 0
            done = False
            steps = 0

            while not done and steps < 1000:
                action_idx = agent.select_action(state)
                move_str = ACTIONS[action_idx]
                moved, reward, done = env.step(move_str)
                next_state = env.get_state().reshape(4,4)

                agent.store_experience(state, action_idx, reward, next_state, done)
                agent.update()

                if agent.steps_done % agent.target_update == 0:
                    agent.update_target()

                state = next_state
                episode_reward += reward
                steps += 1

            if episode_reward > best_score:
                best_score = episode_reward

            results_queue.put(("progress", instance_index, episode, episode_reward))
        results_queue.put(("done", instance_index))

def start_training():
    global all_results
    all_results = {}
    for i, label in instance_progress.items():
        if isinstance(i, int):
            label.config(text=f"Instance {i+1} - Waiting for updates...")

    num_instances = int(expectimax_entry.get())
    num_episodes = int(episode_entry.get())
    selected_method = training_method_var.get()
    summary_label.config(text=f"Training {num_instances} instances with {selected_method}...")

    for i in range(num_instances):
        p = Process(target=run_training_instance, args=(i, num_episodes, results_queue, selected_method))
        p.start()

    open_progress_window(num_instances)

def aggregate_results():
    best_score = 0
    best_instance = -1
    best_episode = -1

    for instance_index, episodes in all_results.items():
        for episode, score in episodes.items():
            if score > best_score:
                best_score = score
                best_instance = instance_index
                best_episode = episode

    os.makedirs("GUI_2048/results", exist_ok=True)
    with open("GUI_2048/results/best_results.pkl", "wb") as f:
        pickle.dump({"best_instance": best_instance, "best_episode": best_episode, "score": best_score}, f)

    summary_label.config(text=f"Best Score: {best_score} (Instance: {best_instance}, Episode: {best_episode})")

def run_best_sequence():
    try:
        with open("GUI_2048/results/best_results.pkl", "rb") as f:
            best_results = pickle.load(f)
    except FileNotFoundError:
        summary_label.config(text="Error: No saved best results found.")
        return

    screen, grid_size, tile_size, base_colors, score_area_height = initialize_game()
    tiles = Tiles(tile_size, score_area_height)
    env = Game2048Env(tiles)
    agent = ExpectimaxAgent(env, max_depth=3)

    running = True
    state = env.reset()

    while running:
        best_move = agent.get_best_move(state)
        if best_move is None:
            print("Game Over!")
            running = False
            break

        moved, _, done = env.step(best_move)
        state = env.get_state().reshape(4,4)

        screen.fill(base_colors["background"])
        draw_grid(screen, grid_size, tile_size, base_colors, score_area_height)
        tiles.draw_tiles(screen)

        score = tiles.score
        font = pygame.font.Font(None, 40)
        score_text = font.render(f"Score: {score}", True, (119, 110, 101))
        screen.blit(score_text, (10, 10))
        pygame.display.flip()

        pygame.time.wait(500)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        if done:
            print("Game Over!")
            running = False

    summary_label.config(text=f"Game Over! Final Score: {tiles.score}")
    pygame.quit()

def setup_gui():
    global root, summary_label, expectimax_entry, episode_entry, instance_progress, training_method_var

    root = tk.Tk()
    root.title("2048 AI and Game Launcher")
    instance_progress = {}

    tk.Label(root, text="Number of Instances:").pack()
    expectimax_entry = tk.Entry(root)
    expectimax_entry.pack()
    expectimax_entry.insert(0, "1")

    tk.Label(root, text="Number of Episodes:").pack()
    episode_entry = tk.Entry(root)
    episode_entry.pack()
    episode_entry.insert(0, "10")

    tk.Label(root, text="Training Method:").pack()
    training_method_var = tk.StringVar(value="expectimax")
    tk.Radiobutton(root, text="Expectimax", variable=training_method_var, value="expectimax").pack()
    tk.Radiobutton(root, text="DQN (Q-learning)", variable=training_method_var, value="dqn").pack()

    summary_label = tk.Label(root, text="")
    summary_label.pack()

    tk.Button(root, text="Play Game", command=run_2048_game).pack()
    tk.Button(root, text="Start Training", command=start_training).pack()
    tk.Button(root, text="Aggregate Results", command=aggregate_results).pack()
    tk.Button(root, text="Run Best Sequence", command=run_best_sequence).pack()

    root.mainloop()

if __name__ == "__main__":
    setup_gui()

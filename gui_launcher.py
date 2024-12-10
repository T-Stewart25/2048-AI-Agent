# gui_launcher.py

import tkinter as tk
from multiprocessing import Process, Queue
import pickle
import os
import numpy as np
import pygame
import torch
import time

from game_2048.game_environment import Game2048Env
from game_2048.tiles import Tiles
from game_2048.setup import initialize_game, draw_grid
from agent.agent import ExpectimaxAgent  # Assuming you have this
from agent.dqn_agent import DQNAgent, ACTIONS, state_to_tensor
from game_2048.play_game import run_2048_game  # If applicable

# Initialize the results queue and data structures
results_queue = Queue()
all_results = {}
progress_window = None
instance_progress = {}

# File paths for storing results and best parameters
SCOREBOARD_FILE = "GUI_2048/results/scoreboard.pkl"
BEST_PARAMS_FILE = "GUI_2048/hyperparams/best_params.pkl"

def load_scoreboard():
    """
    Loads the scoreboard from a pickle file.
    """
    if os.path.exists(SCOREBOARD_FILE):
        with open(SCOREBOARD_FILE, "rb") as f:
            return pickle.load(f)
    else:
        return {
            "best_player_score": 0,
            "best_expectimax_score": 0,
            "best_dqn_score": 0
        }

def save_scoreboard(scoreboard):
    """
    Saves the scoreboard to a pickle file.
    """
    with open(SCOREBOARD_FILE, "wb") as f:
        pickle.dump(scoreboard, f)

def update_scoreboard_display():
    """
    Updates the scoreboard display on the GUI.
    """
    scoreboard = load_scoreboard()
    scores_text = (f"Best Player Score: {scoreboard['best_player_score']} | "
                   f"Best Expectimax Score: {scoreboard['best_expectimax_score']} | "
                   f"Best DQN Score: {scoreboard['best_dqn_score']}")
    scoreboard_label.config(text=scores_text)

def update_progress():
    """
    Updates the progress window with training updates from the queue.
    """
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
            instance_progress[instance_index].config(
                text=f"Training for Instance {instance_index+1} complete. Waiting for others..."
            )

    num_instances = int(expectimax_entry.get())
    num_episodes = int(episode_entry.get())
    if len(all_results) == num_instances and all(
        len(all_results[i]) == num_episodes for i in all_results
    ):
        if progress_window:
            progress_window.after(1000, progress_window.destroy)
            summary_label.config(text="Training done. You may now Aggregate Results.")
    else:
        if progress_window:
            progress_window.after(1000, update_progress)

def open_progress_window(num_instances):
    """
    Opens a new window to display training progress.
    """
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

def run_training_instance(instance_index, num_episodes, results_queue, method, dqn_params=None, expectimax_params=None):
    """
    Runs a single training instance for either Expectimax or DQN.
    """
    game = Tiles()
    env = Game2048Env(game)

    if method == "expectimax":
        max_depth = expectimax_params.get('max_depth', 3)
        agent = ExpectimaxAgent(env, max_depth=max_depth)
        for episode in range(num_episodes):
            state = env.reset()
            done = False
            while not done:
                best_move = agent.get_best_move(state)
                if best_move is None:
                    break
                moved, reward, done = env.step(best_move)
                state = env.get_state().reshape(4,4)
            final_score = env.get_score()
            results_queue.put(("progress", instance_index, episode, final_score))
        results_queue.put(("done", instance_index))

    elif method == "dqn":
        # Extract params or use defaults
        gamma = dqn_params.get('gamma', 0.99)
        lr = dqn_params.get('lr', 0.001)
        batch_size = dqn_params.get('batch_size', 64)
        max_mem = dqn_params.get('max_mem', 10000)
        eps_start = dqn_params.get('eps_start', 1.0)
        eps_end = dqn_params.get('eps_end', 0.01)
        eps_decay = dqn_params.get('eps_decay', 10000)
        target_update = dqn_params.get('target_update', 1000)

        agent = DQNAgent(gamma=gamma, lr=lr, batch_size=batch_size, max_mem=max_mem,
                         eps_start=eps_start, eps_end=eps_end, eps_decay=eps_decay,
                         target_update=target_update)

        os.makedirs("GUI_2048/results", exist_ok=True)
        for episode in range(num_episodes):
            env.reset()
            state = env.get_state().reshape(4,4)
            state = state_to_tensor(state)  # Convert to Tensor
            done = False
            steps = 0

            while not done and steps < 1000:
                action_idx = agent.select_action(state)
                move_str = ACTIONS[action_idx]
                moved, reward, done = env.step(move_str)
                next_state = env.get_state().reshape(4,4)
                next_state = state_to_tensor(next_state)  # Convert to Tensor
                agent.store_experience(state, action_idx, reward, next_state, done)
                agent.update()

                if agent.steps_done % agent.target_update == 0:
                    agent.update_target()

                state = next_state
                steps += 1

            # Save the model after each episode
            torch.save(agent.policy_net.state_dict(), f"GUI_2048/results/model_instance_{instance_index}_episode_{episode}.pth")

            final_score = env.get_score()
            results_queue.put(("progress", instance_index, episode, final_score))
        results_queue.put(("done", instance_index))

def evaluate_model(instance_index, episode, method, num_games, dqn_params=None, expectimax_params=None):
    """
    Evaluates a trained model by running it against the environment for a number of games.
    Returns the average score.
    """
    if dqn_params is None:
        dqn_params = {}

    if expectimax_params is None:
        expectimax_params = {}

    scores = []
    for _ in range(num_games):
        game = Tiles()
        env = Game2048Env(game)

        if method == "expectimax":
            max_depth = expectimax_params.get('max_depth', 3)
            agent = ExpectimaxAgent(env, max_depth=max_depth)
        elif method == "dqn":
            gamma = dqn_params.get('gamma', 0.99)
            lr = dqn_params.get('lr', 0.001)
            batch_size = dqn_params.get('batch_size', 64)
            max_mem = dqn_params.get('max_mem', 10000)
            eps_start = dqn_params.get('eps_start', 1.0)
            eps_end = dqn_params.get('eps_end', 0.01)
            eps_decay = dqn_params.get('eps_decay', 10000)
            target_update = dqn_params.get('target_update', 1000)

            agent = DQNAgent(gamma=gamma, lr=lr, batch_size=batch_size, max_mem=max_mem,
                             eps_start=eps_start, eps_end=eps_end, eps_decay=eps_decay,
                             target_update=target_update)
            model_path = f"GUI_2048/results/model_instance_{instance_index}_episode_{episode}.pth"
            agent.policy_net.load_state_dict(torch.load(model_path, map_location='cpu'))
            agent.policy_net.eval()

        state = env.reset()
        done = False
        steps = 0
        while not done and steps < 1000:
            if method == "expectimax":
                best_move = agent.get_best_move(state)
                if best_move is None:
                    break
                moved, reward, done = env.step(best_move)
            elif method == "dqn":
                with torch.no_grad():
                    q_values = agent.policy_net(state_to_tensor(state))
                    action_idx = q_values.argmax(dim=1).item()
                move_str = ACTIONS[action_idx]
                moved, reward, done = env.step(move_str)

            state = env.get_state().reshape(4,4)
            final_score = env.get_score()
            steps += 1
        scores.append(final_score)
    return np.mean(scores)

def aggregate_results():
    """
    Aggregates training results to identify the best-performing models.
    """
    summary_label.config(text="Aggregating results, please wait, you will be prompted when complete")
    root.update()
    time.sleep(0.5)

    selected_method = training_method_var.get()

    try:
        top_n = int(top_n_entry.get())
        eval_games = int(eval_games_entry.get())
    except ValueError:
        summary_label.config(text="Invalid input for top N or eval games.")
        return

    episodes_list = []
    for instance_index, episodes_dict in all_results.items():
        for episode, score in episodes_dict.items():
            episodes_list.append((score, instance_index, episode))

    # Sort episodes based on score in descending order
    episodes_list.sort(key=lambda x: x[0], reverse=True)
    top_candidates = episodes_list[:top_n]

    best_avg = -float('inf')
    best_candidate = None
    for (score, instance_index, episode) in top_candidates:
        avg_score = evaluate_model(instance_index, episode, selected_method, eval_games)
        if avg_score > best_avg:
            best_avg = avg_score
            best_candidate = (instance_index, episode)

    if best_candidate is None:
        summary_label.config(text="No candidates found for aggregation.")
        return

    best_instance, best_episode = best_candidate
    best_score = best_avg
    os.makedirs("GUI_2048/results", exist_ok=True)
    with open("GUI_2048/results/best_results.pkl", "wb") as f:
        pickle.dump({"best_instance": best_instance, "best_episode": best_episode, "score": best_score}, f)

    summary_label.config(text=f"Aggregation complete. Best Model: Instance {best_instance}, Episode {best_episode}, Avg Score: {best_score:.2f}")

def run_best_sequence():
    """
    Runs the best-performing trained model in the game GUI.
    """
    try:
        with open("GUI_2048/results/best_results.pkl", "rb") as f:
            best_results = pickle.load(f)
    except FileNotFoundError:
        summary_label.config(text="Error: No saved best results found.")
        return

    selected_method = training_method_var.get()

    screen, grid_size, tile_size, base_colors, score_area_height = initialize_game()
    tiles = Tiles(tile_size, score_area_height)
    env = Game2048Env(tiles)

    # Load best params if available and if use_best_params is checked
    dqn_params = {}
    expectimax_params = {}
    if use_best_params_var.get() == 1 and os.path.exists(BEST_PARAMS_FILE):
        with open(BEST_PARAMS_FILE, "rb") as f:
            p_data = pickle.load(f)
            if p_data['method'] == selected_method and p_data['params'] is not None:
                if selected_method == 'expectimax':
                    expectimax_params = p_data['params']
                elif selected_method == 'dqn':
                    dqn_params = p_data['params']

    if selected_method == "expectimax":
        max_depth = expectimax_params.get('max_depth', 3)
        agent = ExpectimaxAgent(env, max_depth=max_depth)
    elif selected_method == "dqn":
        agent = DQNAgent(**dqn_params)
        best_instance = best_results['best_instance']
        best_episode = best_results['best_episode']
        model_path = f"GUI_2048/results/model_instance_{best_instance}_episode_{best_episode}.pth"
        agent.policy_net.load_state_dict(torch.load(model_path, map_location='cpu'))
        agent.policy_net.eval()

    running = True
    state = env.reset()
    final_score = 0

    while running:
        if selected_method == "expectimax":
            best_move = agent.get_best_move(state)
            if best_move is None:
                print("Game Over!")
                running = False
                break
            moved, _, done = env.step(best_move)
        elif selected_method == "dqn":
            with torch.no_grad():
                q_values = agent.policy_net(state_to_tensor(state))
                action_idx = q_values.argmax(dim=1).item()
            move_str = ACTIONS[action_idx]
            moved, _, done = env.step(move_str)

        state = env.get_state().reshape(4,4)
        final_score = env.get_score()

        screen.fill(base_colors["background"])
        draw_grid(screen, grid_size, tile_size, base_colors, score_area_height)
        tiles.draw_tiles(screen)

        font = pygame.font.Font(None, 40)
        score_text = font.render(f"Score: {final_score}", True, (119, 110, 101))
        screen.blit(score_text, (10, 10))
        pygame.display.flip()
        pygame.time.wait(500)  # Wait for half a second to visualize the move

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        if done:
            print("Game Over!")
            running = False

    summary_label.config(text=f"Game Over! Final Score: {final_score}")
    pygame.quit()

def run_player_game_from_gui():
    """
    Allows the user to play the 2048 game manually via the GUI.
    """
    screen, grid_size, tile_size, base_colors, score_area_height = initialize_game()
    tiles = Tiles(tile_size, score_area_height)
    
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_DOWN:
                    tiles.move('down')
                elif event.key == pygame.K_UP:
                    tiles.move('up')
                elif event.key == pygame.K_LEFT:
                    tiles.move('left')
                elif event.key == pygame.K_RIGHT:
                    tiles.move('right')

        if tiles.check_game_over():
            print("Game Over!")
            running = False

        screen.fill(base_colors['background'])
        draw_grid(screen, grid_size, tile_size, base_colors, score_area_height)
        tiles.draw_tiles(screen)

        current_score = tiles.get_score()
        font = pygame.font.Font(None, 40)
        score_text = font.render(f"Score: {current_score}", True, (119, 110, 101))
        screen.blit(score_text, (10, 10))
        pygame.display.flip()

    pygame.quit()

    # Update the scoreboard if the player's score is a new high
    scoreboard = load_scoreboard()
    if current_score > scoreboard['best_player_score']:
        scoreboard['best_player_score'] = current_score
        save_scoreboard(scoreboard)
    update_scoreboard_display()
    summary_label.config(text=f"Game Over! Player Score: {current_score}")

def run_hyperparam_search():
    """
    Performs a hyperparameter search to find the best-performing parameters.
    """
    summary_label.config(text="Running hyperparameter search, please wait...")
    root.update()
    time.sleep(0.5)

    selected_method = training_method_var.get()

    # Define parameter grids
    # For Expectimax
    expectimax_depth_values = [3, 4, 5]

    # For DQN
    dqn_gamma_values = [0.95, 0.99]
    dqn_lr_values = [0.001, 0.0005]
    dqn_eps_decay_values = [5000, 10000]

    num_instances_test = 1
    num_episodes_test = 5
    best_param_score = -float('inf')
    best_params = None

    os.makedirs("GUI_2048/hyperparams", exist_ok=True)

    if selected_method == "expectimax":
        for depth in expectimax_depth_values:
            local_queue = Queue()
            all_results_test = {}
            p = Process(target=run_training_instance,
                        args=(0, num_episodes_test, local_queue, 'expectimax', None, {'max_depth': depth}))
            p.start()

            episodes_run = 0
            while episodes_run < num_episodes_test:
                msg = local_queue.get()
                if msg[0] == "progress":
                    _, inst_i, ep_i, sc_i = msg
                    if inst_i not in all_results_test:
                        all_results_test[inst_i] = {}
                    all_results_test[inst_i][ep_i] = sc_i
                elif msg[0] == "done":
                    episodes_run = num_episodes_test
            p.join()

            # Evaluate the best score from this depth
            best_local = max(all_results_test[0].values())
            if best_local > best_param_score:
                best_param_score = best_local
                best_params = {'max_depth': depth}

    elif selected_method == "dqn":
        # Try all combinations
        for gamma_val in dqn_gamma_values:
            for lr_val in dqn_lr_values:
                for eps_decay_val in dqn_eps_decay_values:
                    dqn_params_test = {
                        'gamma': gamma_val,
                        'lr': lr_val,
                        'eps_decay': eps_decay_val
                    }
                    local_queue = Queue()
                    all_results_test = {}
                    p = Process(target=run_training_instance,
                                args=(0, num_episodes_test, local_queue, 'dqn', dqn_params_test, None))
                    p.start()

                    episodes_run = 0
                    while episodes_run < num_episodes_test:
                        msg = local_queue.get()
                        if msg[0] == "progress":
                            _, inst_i, ep_i, sc_i = msg
                            if inst_i not in all_results_test:
                                all_results_test[inst_i] = {}
                            all_results_test[inst_i][ep_i] = sc_i
                        elif msg[0] == "done":
                            episodes_run = num_episodes_test

                    p.join()

                    # Evaluate the best score from this parameter set
                    best_local = max(all_results_test[0].values())
                    if best_local > best_param_score:
                        best_param_score = best_local
                        best_params = {
                            'gamma': gamma_val,
                            'lr': lr_val,
                            'eps_decay': eps_decay_val
                        }

    # Save the best parameters found
    with open(BEST_PARAMS_FILE, "wb") as f:
        pickle.dump({'method': selected_method, 'params': best_params, 'score': best_param_score}, f)

    summary_label.config(text=f"Hyperparameter search complete. Best params: {best_params}, Score: {best_param_score}")

def start_training():
    """
    Starts the training process based on user-selected parameters and method.
    """
    global all_results
    all_results = {}
    for i, label in instance_progress.items():
        if isinstance(i, int):
            label.config(text=f"Instance {i+1} - Waiting for updates...")

    num_instances = int(expectimax_entry.get())
    num_episodes = int(episode_entry.get())
    selected_method = training_method_var.get()

    # If "Use Best Params" is checked and file exists, load them
    dqn_params = {}
    expectimax_params = {}
    if use_best_params_var.get() == 1 and os.path.exists(BEST_PARAMS_FILE):
        with open(BEST_PARAMS_FILE, "rb") as f:
            p_data = pickle.load(f)
            if p_data['method'] == selected_method and p_data['params'] is not None:
                if selected_method == 'expectimax':
                    expectimax_params = p_data['params']
                elif selected_method == 'dqn':
                    dqn_params = p_data['params']

    summary_label.config(text=f"Training {num_instances} instances with {selected_method}...")
    for i in range(num_instances):
        p = Process(target=run_training_instance, args=(i, num_episodes, results_queue, selected_method, dqn_params, expectimax_params))
        p.start()

    open_progress_window(num_instances)

def setup_gui():
    """
    Sets up the Tkinter GUI for the application.
    """
    global root, summary_label, expectimax_entry, episode_entry, instance_progress, training_method_var, scoreboard_label, top_n_entry, eval_games_entry
    global use_best_params_var

    root = tk.Tk()
    root.title("2048 AI and Game Launcher")
    instance_progress = {}

    # Number of Instances
    tk.Label(root, text="Number of Instances:").pack()
    expectimax_entry = tk.Entry(root)
    expectimax_entry.pack()
    expectimax_entry.insert(0, "1")

    # Number of Episodes
    tk.Label(root, text="Number of Episodes:").pack()
    episode_entry = tk.Entry(root)
    episode_entry.pack()
    episode_entry.insert(0, "10")

    # Training Method Selection
    tk.Label(root, text="Training Method:").pack()
    training_method_var = tk.StringVar(value="expectimax")
    tk.Radiobutton(root, text="Expectimax", variable=training_method_var, value="expectimax").pack()
    tk.Radiobutton(root, text="DQN (Q-learning)", variable=training_method_var, value="dqn").pack()

    # Top N Models for Aggregation
    tk.Label(root, text="Top N models to select:").pack()
    top_n_entry = tk.Entry(root)
    top_n_entry.pack()
    top_n_entry.insert(0, "100")

    # Evaluation Games per Model
    tk.Label(root, text="Evaluation games per model:").pack()
    eval_games_entry = tk.Entry(root)
    eval_games_entry.pack()
    eval_games_entry.insert(0, "20")

    # Use Best Parameters Checkbox
    use_best_params_var = tk.IntVar()
    tk.Checkbutton(root, text="Use Best Parameters", variable=use_best_params_var).pack()

    # Summary Label
    summary_label = tk.Label(root, text="")
    summary_label.pack()

    # Buttons for Various Actions
    tk.Button(root, text="Play Game", command=run_player_game_from_gui).pack()
    tk.Button(root, text="Start Training", command=start_training).pack()
    tk.Button(root, text="Aggregate Results", command=aggregate_results).pack()
    tk.Button(root, text="Run Best Sequence", command=run_best_sequence).pack()
    tk.Button(root, text="Find Parameters", command=run_hyperparam_search).pack()

    # Scoreboard Display
    scoreboard_label = tk.Label(root, text="", fg="blue")
    scoreboard_label.pack()

    # Initialize the scoreboard display
    update_scoreboard_display()

    # Start the Tkinter main loop
    root.mainloop()

if __name__ == "__main__":
    setup_gui()

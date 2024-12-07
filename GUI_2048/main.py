import tkinter as tk
from multiprocessing import Process, Queue
from game_2048.main import run_2048_game
from agent.agent import run_expectimax_agent
from game_2048.setup import initialize_game, draw_grid
import pickle
import os
import time
from game_2048.setup import initialize_game, draw_grid
from game_2048.tiles import Tiles
from game_2048.game_environment import Game2048Env
from agent.agent import ExpectimaxAgent
import pygame

# Global dictionary to store scores for all instances and episodes
all_results = {}
progress_window = None
results_queue = Queue()

def open_progress_window(num_instances):
    """Open a single window to display live progress updates for each Expectimax instance."""
    global progress_window
    progress_window = tk.Toplevel(root)
    progress_window.title("Expectimax Training Progress")

    # Initialize labels for each instance's progress
    for i in range(num_instances):
        instance_progress[i] = tk.Label(progress_window, text=f"Instance {i} - Waiting for updates...")
        instance_progress[i].pack()

    # Final completion message label
    instance_progress["final"] = tk.Label(progress_window, text="Training in progress...")
    instance_progress["final"].pack()

    # Start updating progress in the progress window
    if progress_window:
        progress_window.after(1000, lambda: update_progress())

def update_progress():
    """Periodically update the progress window."""
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
                text=f"Training for Instance {instance_index} complete. Waiting for others..."
            )

    # Check if training is complete for all instances
    num_instances = int(expectimax_entry.get())
    num_episodes = int(episode_entry.get())
    if len(all_results) == num_instances and all(
        len(all_results[i]) == num_episodes for i in all_results
    ):
        if progress_window:
            progress_window.after(1000, lambda: progress_window.destroy())
            summary_label.config(text="Training done. Press 'Aggregate Results' to see the best model.")
    else:
        if progress_window:
            progress_window.after(1000, update_progress)

def start_expectimax_training():
    """Start the Expectimax training processes."""
    global all_results
    all_results = {}  # Reset results for a new training run

    # Reset progress display
    for i, label in instance_progress.items():
        if isinstance(i, int):
            label.config(text=f"Instance {i} - Waiting for updates...")

    # Display the training status in the main GUI
    num_instances = int(expectimax_entry.get())
    num_episodes = int(episode_entry.get())
    summary_label.config(text=f"Training {num_instances} instances...")

    # Start each Expectimax instance
    for i in range(num_instances):
        p = Process(target=run_training_instance, args=(i, num_episodes, results_queue))
        p.start()

    # Open the progress window
    open_progress_window(num_instances)

def run_training_instance(instance_index, num_episodes, results_queue):
    """Run Expectimax training for a single instance."""
    from game_2048.tiles import Tiles
    from game_2048.game_environment import Game2048Env
    from agent.agent import ExpectimaxAgent

    game = Tiles()
    env = Game2048Env(game)
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

        # Send progress update to the main process
        results_queue.put(("progress", instance_index, episode, total_score))

    # Notify main process of completion
    results_queue.put(("done", instance_index))

def aggregate_results():
    """Aggregate the results to find the best-performing model."""
    best_score = 0
    best_instance = -1
    best_episode = -1

    for instance_index, episodes in all_results.items():
        for episode, score in episodes.items():
            if score > best_score:
                best_score = score
                best_instance = instance_index
                best_episode = episode

    # Save the best results
    os.makedirs("GUI_2048/results", exist_ok=True)
    with open("GUI_2048/results/best_results.pkl", "wb") as f:
        pickle.dump({"best_instance": best_instance, "best_episode": best_episode, "score": best_score}, f)

    summary_label.config(text=f"Best Score: {best_score} (Instance: {best_instance}, Episode: {best_episode})")

def run_best_sequence():
    """Run the game using the best-performing model dynamically with consistent visuals."""
    try:
        with open("GUI_2048/results/best_results.pkl", "rb") as f:
            best_results = pickle.load(f)
    except FileNotFoundError:
        summary_label.config(text="Error: No saved best results found.")
        return

    # Initialize game components
    screen, grid_size, tile_size, base_colors, score_area_height = initialize_game()
    tiles = Tiles(tile_size, score_area_height)
    env = Game2048Env(tiles)
    agent = ExpectimaxAgent(env, max_depth=3)

    # AI-controlled game loop
    running = True
    state = env.reset()

    while running:
        # Let the AI decide the best move
        best_move = agent.get_best_move(state)
        if best_move is None:  # No valid moves, game over
            print("Game Over!")
            running = False
            break

        # Perform the move and update the state
        moved, _, done = env.step(best_move)
        if done:
            print("Game Over!")
            running = False

        # Update the Pygame screen
        screen.fill(base_colors["background"])
        draw_grid(screen, grid_size, tile_size, base_colors, score_area_height)
        tiles.draw_tiles(screen)

        # Display the current score
        score = tiles.score
        font = pygame.font.Font(None, 40)
        score_text = font.render(f"Score: {score}", True, (119, 110, 101))
        screen.blit(score_text, (10, 10))
        pygame.display.flip()

        # Add a short delay to control the pace of gameplay
        pygame.time.wait(500)

        # Handle Pygame events to allow closing the game window
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

    # Final Game Over Display
    summary_label.config(text=f"Game Over! Final Score: {tiles.score}")
    pygame.quit()



def setup_gui():
    """Set up the GUI for the application."""
    global root, summary_label, expectimax_entry, episode_entry, instance_progress

    root = tk.Tk()
    root.title("2048 AI and Game Launcher")
    instance_progress = {}

    tk.Label(root, text="Number of Expectimax Instances:").pack()
    expectimax_entry = tk.Entry(root)
    expectimax_entry.pack()
    expectimax_entry.insert(0, "1")

    tk.Label(root, text="Number of Episodes:").pack()
    episode_entry = tk.Entry(root)
    episode_entry.pack()
    episode_entry.insert(0, "10")

    summary_label = tk.Label(root, text="")
    summary_label.pack()

    tk.Button(root, text="Play Game", command=run_2048_game).pack()
    tk.Button(root, text="Start Expectimax Training", command=start_expectimax_training).pack()
    tk.Button(root, text="Aggregate Results", command=aggregate_results).pack()
    tk.Button(root, text="Run Best Sequence", command=run_best_sequence).pack()

    root.mainloop()

if __name__ == "__main__":
    setup_gui()

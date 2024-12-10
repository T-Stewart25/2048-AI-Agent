# **2048 AI Project**

An advanced implementation of the 2048 game, featuring **Expectimax** and **Deep Q-Network (DQN)** AI agents. This project includes a Tkinter-based GUI launcher that allows users to interact with the game, train agents, and evaluate their performance.

Developed entirely in Python, the project showcases multiple functionalities such as manual gameplay, hyperparameter tuning, multi-threaded training, results aggregation, and real-time demonstrations of the best-performing AI models. The GUI is designed for an intuitive user experience and provides visual feedback during training and evaluation.

To examine the game specifically, please go to the game repository at [2048 Game](https://github.com/T-Stewart25/2048-Game).

## **Key Features**

- **Expectimax Agent**: Leverages the Expectimax algorithm for strategic decision-making.
- **DQN Agent**: Utilizes a Deep Q-Network with experience replay and target networks for reinforcement learning.
- **User-Friendly GUI**: A Tkinter-based graphical interface for seamless interaction with the game and AI agents.
- **Multi-Threaded Training**: Supports simultaneous training of multiple models, with each running on its own thread for optimized performance.
- **Progress Monitoring**: Opens a progress window during training, showing real-time updates for all threads.
- **Hyperparameter Tuning**: Allows users to input training parameters such as the number of instances (threads) and episodes for fine-tuning performance.
- **Results Aggregation**: Aggregates top-performing models based on user-defined criteria and tests them against multiple games to select the best one.
- **Real-Time AI Demonstration**: Visually showcases the best model playing the game in real time.
- **High Score Tracking**: Tracks and saves the highest scores for manual gameplay, Expectimax, and DQN models.

---

## **Table of Contents**

- [Features](#key-features)
- [Demo](#demo)
- [Installation](#installation)
- [Usage](#usage)
  - [Playing Manually](#playing-manually)
  - [Training Agents](#training-agents)
  - [Evaluating Models](#evaluating-models)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

---

## **Demo**

![Gameplay Demo](https://your-image-link.com/demo.gif)

*Caption: Demonstration of the Expectimax and DQN agents in action.*

---

## **Installation**

### **Prerequisites**
- Python 3.7 or higher
- [Pip](https://pip.pypa.io/en/stable/) package manager

### **Setup Instructions**

1. **Clone the Repository**
   ```bash
   git clone https://github.com/your-username/2048-AI-Project.git
   cd 2048-AI-Project
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the GUI Launcher**
   ```bash
   python gui_launcher.py
   ```

---

## **Usage**

### **Playing Manually**
- Launch the GUI and play the game manually using the arrow keys.
- The game will display your score and allow you to try reaching the 2048 tile.

### **Training Agents**
1. Open the GUI.
2. Enter the number of **instances** (threads) and **episodes** for training:
   - Recommended: 30 instances, 5000 episodes (adjust based on your machine's capability).
3. Select the AI model to train (**Expectimax** or **DQN**).
4. Begin the training process:
   - Click **Start Training** to train the models.
   - Progress for each thread will be displayed in a separate pop-up window.

### **Evaluating Models**
1. After training completes, models are automatically saved.
2. Choose the number of models to test during the **aggregation** stage.
3. Specify the number of games each model should play.
4. Click **Aggregate Results** to determine the best-performing model.
5. View the top-performing model in action by selecting **Run Best Sequence**.

---

## **Project Structure**

```
2048-AI-Project/
├── agent/
│   ├── agent.py                  # Expectimax AI agent
│   ├── dqn_agent.py              # Deep Q-Network agent
├── game_2048/
│   ├── environment.py            # Game environment logic
│   ├── tiles.py                  # Tile management
│   ├── setup.py                  # GUI setup
├── GUI_2048/
│   ├── hyperparams               # Saved Hyperparameter Values
│   └──  models/                  # Saved AI models
├── gui_launcher.py               # Main GUI script
├── README.md                     # Project documentation
├── requirements.txt              # Python dependencies
└── LICENSE                       # License information
```

---

## **Contributing**

Contributions are welcome! Here's how you can get involved:
1. Fork the repository.
2. Create a feature branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. Commit your changes:
   ```bash
   git commit -m "Add your feature description"
   ```
4. Push to your branch:
   ```bash
   git push origin feature/your-feature-name
   ```
5. Submit a pull request for review.

---

## **License**

This project is licensed under the [MIT License](LICENSE). Feel free to use, modify, and distribute this project under the terms of the license.

---

## **Contact**

- **Author**: Thomas Stewart
- **Email**: thomaslstewart1@gmail.com
- **GitHub**: [T-Stewart25](https://github.com/T-Stewart25)
- **Personal Portfolio**: [Portfolio](https://thomasstewartpersonal.com)

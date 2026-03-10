# 🤖 Reinforcement Learning Project

## 📌 Project Overview

This project focuses on implementing core **Reinforcement Learning (RL)** algorithms and applying them to different environments. The objective is to design agents that learn optimal behavior through interaction with their environment.

Two fundamental reinforcement learning algorithms are implemented:

* **Value Iteration**
* **Q-Learning**

These agents are first tested in a **Gridworld environment**, then applied to more complex systems including a **simulated robot controller (Crawler)** and the **Pacman game environment**.

The project demonstrates how agents can learn optimal policies using **Markov Decision Processes (MDPs)** and experience-based learning.

---

# 🧠 Implemented Algorithms

## 1️⃣ Value Iteration

Value Iteration is a **dynamic programming algorithm** used to compute the optimal policy for a known **Markov Decision Process (MDP)**.

The algorithm repeatedly updates state values using the Bellman equation:

[
V(s) = \max_a \sum_{s'} P(s'|s,a) [R(s,a,s') + \gamma V(s')]
]

Key steps:

1. Initialize state values.
2. Iteratively update values using expected future rewards.
3. Extract the optimal policy from the final values.

Value Iteration assumes that the **environment model is known**.

---

## 2️⃣ Q-Learning

Q-Learning is a **model-free reinforcement learning algorithm** that allows agents to learn optimal policies through experience.

Instead of computing state values directly, Q-learning estimates the value of **state-action pairs**:

[
Q(s,a) \leftarrow Q(s,a) + \alpha \left[ r + \gamma \max_{a'} Q(s',a') - Q(s,a) \right]
]

Key features:

* No prior knowledge of the environment is required.
* Agents learn by interacting with the environment.
* Policies are improved over time using exploration and exploitation.

---

# 🎮 Environments Used

## Gridworld

Gridworld is a simple grid-based environment used to test reinforcement learning algorithms.

The agent moves through a grid where:

* Each cell represents a state
* Actions move the agent in different directions
* Some states provide rewards or penalties

This environment is used to validate the correctness of **Value Iteration and Q-Learning implementations**.

---

## Crawler (Simulated Robot)

The **Crawler** is a simulated robot controller where reinforcement learning is used to teach a robot how to move forward.

The agent learns:

* How to control its joints
* How to coordinate movement
* How to maximize forward motion

This demonstrates RL applied to **robotic control problems**.

---

## Pacman

Finally, reinforcement learning is applied to the **Pacman environment**, where Pacman learns strategies through experience rather than explicit programming.

The agent learns to:

* Collect food efficiently
* Avoid ghosts
* Maximize score over time


# 📄 Important Supporting Files

### mdp.py

Defines the interface for **Markov Decision Processes**.

---

### learningAgents.py

Defines base classes including:

* `ValueEstimationAgent`
* `QLearningAgent`

Your agents will extend these classes.

---

### util.py

Contains helper functions and useful data structures, including **util.Counter**, which is commonly used for storing Q-values.

---

### gridworld.py

Implements the **Gridworld environment** used for testing reinforcement learning agents.

---

### featureExtractors.py

Defines feature extraction methods used for **approximate Q-learning**.

---

# ▶️ Running the Autograder

To grade all questions:

```
python autograder.py
```

To grade a specific question:

```
python autograder.py -q q2
```

To run a specific test case:

```
python autograder.py -t test_cases/q2/1-bridge-grid
```

The autograder evaluates whether your implementations produce the correct behavior.


# 👨‍💻 Author

**Student:** Nguyen Thi Hong Hanh
**Course:** Artificial Intelligence


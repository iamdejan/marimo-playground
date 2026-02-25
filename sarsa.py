import marimo

__generated_with = "0.20.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Sarsa Algorithm
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Initialize environment
    """)
    return


@app.cell
def _():
    import numpy as np
    from typing import Tuple
    from jaxtyping import jaxtyped
    from beartype import beartype as typechecker

    size = (5, 5)  # (row, column)
    actions: list[Tuple[int, int]] = [
        (-1, 0),  # top
        (0, 1),  # right
        (1, 0),  # bottom
        (0, -1),  # left
        (0, 0),  # stay
    ]

    reward_probability: float = 1.0  # p(r|s,a)
    state_transition_probability: float = 1.0  # p(s'|s,a)
    convergence_threshold: float = 1e-4
    discount_rate: float = 0.9
    alpha: float = 0.05  # learning rate
    epsilon = 0.15  # 0.1 = 10% random actions

    # Initialize reward
    reward_boundary = -10.0
    reward_forbidden = -10.0
    reward_goal: float = 1.0
    goal: Tuple[int, int] = (3, 2)

    forbidden_cells: list[Tuple[int, int]] = [
        (1, 1),
        (1, 2),
        (2, 2),
        (3, 1),
        (3, 3),
        (4, 1),
    ]


    def is_out_of_bounds(r: int, c: int) -> bool:
        return r < 0 or r >= size[0] or c < 0 or c >= size[1]


    def calculate_reward(r: int, c: int) -> int:
        if is_out_of_bounds(r, c):
            return reward_boundary

        # hardcode forbidden cells
        if (r, c) in forbidden_cells:
            return reward_forbidden

        if (r, c) == goal:
            return reward_goal

        return 0

    return (
        actions,
        alpha,
        calculate_reward,
        discount_rate,
        epsilon,
        goal,
        is_out_of_bounds,
        np,
        size,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Search for optimal value and optimal policy
    """)
    return


@app.cell
def _(
    actions: "list[Tuple[int, int]]",
    alpha: float,
    calculate_reward,
    discount_rate: float,
    epsilon,
    goal: "Tuple[int, int]",
    is_out_of_bounds,
    np,
    size,
):
    import math

    seed = 1772030113
    np.random.seed(seed)

    # Initialize value state
    v = np.zeros(shape=(size[0] * size[1],), dtype=np.float64)
    state_space_size = len(v)

    # Initialize q-table
    actions_len = len(actions)
    q = np.zeros(shape=(state_space_size, actions_len), dtype=np.float64)

    # Initialize policy (equal probability for all actions initially)
    policy = np.full(
        shape=(actions_len, state_space_size), fill_value=1.0 / actions_len, dtype=np.float64
    )

    # Episode
    num_episodes = 1000  # Added an outer loop for episodes
    episode_lengths = np.zeros(shape=(num_episodes,), dtype=np.int64)
    rewards = np.zeros(shape=(num_episodes,), dtype=np.int64)

    for episode in range(num_episodes):
        s = 0  # Initialize state from top-left (cell 0) at the start of EVERY episode

        # 1. SAMPLE action based on policy probabilities
        a = np.random.choice(actions_len, p=policy[:, s])

        episode_length = 0
        total_reward = 0
        while True:
            # 2. Fix math: Use size[1] (columns) for coordinate conversion
            r = math.floor(s / size[1])
            c = s % size[1]

            if (r, c) == goal:
                break

            move = actions[a]
            next_r: int = r + move[0]
            next_c: int = c + move[1]
            reward = calculate_reward(next_r, next_c)
            total_reward += reward

            if is_out_of_bounds(next_r, next_c):
                # bounce back
                next_r = r
                next_c = c

            next_s = math.floor(next_r * size[1] + next_c)

            # 3. SAMPLE next action based on probabilities
            # If
            next_a = np.random.choice(actions_len, p=policy[:, next_s])

            # SARSA Q-table update (In-place)
            q[s, a] = q[s, a] - alpha * (q[s, a] - (reward + discount_rate * q[next_s, next_a]))

            # Epsilon-greedy policy update (In-place)
            best_a = np.argmax(q[s, :])
            for a_idx in range(actions_len):
                if a_idx == best_a:
                    # Simplified your math here slightly, does the exact same thing
                    policy[a_idx, s] = 1.0 - epsilon + (epsilon / actions_len)
                else:
                    policy[a_idx, s] = epsilon / actions_len

            # Move to next state/action
            s = next_s
            a = next_a
            episode_length += 1

        # Print progress occasionally
        print(f"Episode {episode + 1} completed")
        episode_lengths[episode] = episode_length
        rewards[episode] = total_reward

    print("Training finished!")
    return episode_lengths, policy, rewards


@app.cell
def _(episode_lengths):
    episode_lengths
    return


@app.cell
def _(np, policy):
    move_symbols = ["↑", "→", "↓", "←", "∘"]


    def print_best_move_on_grid(best_move_according_to_policy):
        i = 0
        while i < len(best_move_according_to_policy):
            for w in range(5):
                best_move_index = best_move_according_to_policy[i]
                print(move_symbols[best_move_index], end="\t")
                i += 1
            print("")


    best_move_according_to_policy = np.argmax(policy, axis=0)
    print_best_move_on_grid(best_move_according_to_policy)
    return


@app.cell
def _(episode_lengths):
    import matplotlib.pyplot as plt


    def render_episode_lengths():
        # Assuming episode_lengths is already defined from your previous cell
        # episode_lengths = np.array([...])

        # Create the figure and axis objects
        fig, ax = plt.subplots(figsize=(10, 5))

        # Plot the raw data
        ax.plot(episode_lengths, color="#1f77b4", alpha=0.8, linewidth=1.5)

        # Add labels and a title
        ax.set_title("SARSA Agent Training Progress", fontsize=14)
        ax.set_xlabel("Episode index", fontsize=12)
        ax.set_ylabel("Episode length", fontsize=12)

        # Add a grid to make it easier to read
        ax.grid(True, linestyle="--", alpha=0.6)
        return fig


    # In Marimo, simply leaving the figure variable at the bottom renders it
    episode_lengths_fig = render_episode_lengths()
    episode_lengths_fig
    return (plt,)


@app.cell
def _(plt, rewards):
    def render_rewards():
        # Assuming episode_lengths is already defined from your previous cell
        # episode_lengths = np.array([...])

        # Create the figure and axis objects
        fig, ax = plt.subplots(figsize=(10, 5))

        # Plot the raw data
        ax.plot(rewards, color="#1f77b4", alpha=0.8, linewidth=1.5)

        # Add labels and a title
        ax.set_title("SARSA Agent Training Rewards", fontsize=14)
        ax.set_xlabel("Episode index", fontsize=12)
        ax.set_ylabel("Totla rewards", fontsize=12)

        # Add a grid to make it easier to read
        ax.grid(True, linestyle="--", alpha=0.6)

        # In Marimo, simply leaving the figure variable at the bottom renders it
        return fig


    rewards_fig = render_rewards()
    rewards_fig
    return


if __name__ == "__main__":
    app.run()

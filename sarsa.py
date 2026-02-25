import marimo

__generated_with = "0.19.11"
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
    alpha: float = 1e-3  # learning rate

    # Initialize reward
    reward_boundary = -1.0
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
    goal: "Tuple[int, int]",
    is_out_of_bounds,
    np,
    size,
):
    import math

    # Initialize value state
    v = np.zeros(shape=(size[0] * size[1],), dtype=np.float64)
    state_space_size = len(v)
    v_history: list[np.ndarray] = []

    # Initialize q-table
    actions_len = len(actions)
    q = np.zeros(shape=(state_space_size, actions_len), dtype=np.float64)
    q_history: list[np.ndarray] = []

    # Initialize policy
    policy = np.full(shape=(actions_len, state_space_size), fill_value=0.2, dtype=np.float64)

    # Initialize epsilon
    epsilon = 1  # 1 = 100% random actions
    epsilon_decay_rate = 0.00001  # epsilon decay rate

    s = 0  # initialize state from top-left (cell 0)
    episode = 0
    a = np.argmax(policy[:, s])
    while True:
        r = math.floor(s / size[0])
        c = s % size[1]
        print(f"current location = {r, c}")
        if (r, c) == goal:
            break

        move = actions[a]
        print(f"a = {a}, move = {move}")
        next_r: int = r + move[0]
        next_c: int = c + move[1]
        reward = calculate_reward(next_r, next_c)
        if is_out_of_bounds(next_r, next_c):
            # bounce back
            next_r = r
            next_c = c
        next_s = math.floor(next_r * size[0] + next_c)
        next_a = np.argmax(policy[:, next_s])

        next_q = q.copy()
        next_q[s, a] = q[s, a] - alpha * (q[s, a] - (reward + discount_rate * q[next_s, next_a]))

        next_policy = policy.copy()
        best_a = np.argmax(next_q[s, :])
        for a_idx in range(len(actions)):
            if a_idx == best_a:
                next_policy[a_idx, s] = 1 - (epsilon / len(actions)) * (len(actions) - 1)
            else:
                next_policy[a_idx, s] = epsilon / len(actions)

        q = next_q
        policy = next_policy
        s = next_s
        a = next_a

        episode += 1
        epsilon = max(epsilon - epsilon_decay_rate, 0.0)
        print("")
    return episode, policy


@app.cell
def _(episode):
    episode
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
def _():
    return


if __name__ == "__main__":
    app.run()

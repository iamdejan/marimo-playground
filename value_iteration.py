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
    # Value Iteration

    Value iteration is the first algorithm that is discussed in [Mathematical Foundations for Machine Learning](https://github.com/MathFoundationRL/Book-Mathematical-Foundation-of-Reinforcement-Learning) book. The algorithm goes like this:
    - Initialization:
        - Define the probability models $p(r|s,a)$ and $p(s'|s,a)$ for all state-action pairs $(s,a)$.
        - Set initial value $v_0$ as $0$ (or any arbitrary number, but for simplicity set to $0$).
    - Goal: Search for the optimal state value and an optimal policy for solving the Bellman optimality equation.

    While $v_k$ has not converged in the sense that $|| v_k - v_{k -1} ||$ is greater than a predefined small threshold, for the $k$th iteration do:<br>
    &emsp;For every state $s \in S$ do<br>
    &emsp;&emsp;For every action $a in A(s)$, do<br>
    &emsp;&emsp;&emsp;q-value: $q(s,a) = \sum_{r} p(r|s,a) r + \gamma \sum_{s'} p(s'|s,a) v_{k}(s')$<br>
    &emsp;&emsp;Maximum q-value: $a_k^*(s) = \text{arg} \underset{a}{\text{max}} \space q_{k}(s,a)$<br>
    &emsp;&emsp;Policy update: $\pi_{k+1}(a|s) = 1$ if $a = a_{k}^{*}$ else $0$<br>
    &emsp;&emsp;Value update: $v_{k+1}(s) = \underset{a}{max} \space q_{k}(s,a)$
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

    size = (2, 2)
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

    # Initialize reward
    reward_boundary = reward_forbidden = -1.0
    reward_goal: float = 1
    goal: Tuple[int, int] = (1, 1)

    forbidden_cells: list[Tuple[int, int]] = [(0, 1)]


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
        calculate_reward,
        convergence_threshold,
        discount_rate,
        is_out_of_bounds,
        np,
        reward_probability,
        size,
        state_transition_probability,
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
    calculate_reward,
    convergence_threshold: float,
    discount_rate: float,
    is_out_of_bounds,
    np,
    reward_probability: float,
    size,
    state_transition_probability: float,
):
    import math

    # Initialize value state
    v: np.ndarray = np.zeros(shape=(size[0] * size[1],), dtype=np.float64)
    v_history: list[np.ndarray] = []
    state_space_size: int = len(v)

    # Initialize q-table
    q = np.zeros(shape=(state_space_size, len(actions)), dtype=np.float64)

    # Initialize policy
    policy_history: list[np.ndarray] = []

    k = 0
    while True:
        new_v = np.zeros(shape=(size[0] * size[1],), dtype=np.float64)
        policy = np.zeros(shape=(len(actions), state_space_size), dtype=np.float64)
        delta = 0
        for s in range(state_space_size):
            r = math.floor(s / size[0])
            c = s % size[1]

            max_q = -float("inf")
            best_a = None
            for a in range(len(actions)):
                move = actions[a]
                next_r = r + move[0]
                next_c = c + move[1]
                immediate_reward = calculate_reward(next_r, next_c)
                if is_out_of_bounds(next_r, next_c):
                    # bounce back
                    next_r = r
                    next_c = c
                next_s = next_r * size[0] + next_c
                v_next_state = v[next_s].item()  # use v_{k}, not v_{k+1}
                q[s, a] = reward_probability * immediate_reward + discount_rate * state_transition_probability * v_next_state

                if q[s, a] > max_q:
                    max_q = q[s, a].item()
                    best_a = a

            policy[best_a, s] = 1
            new_v[s] = max_q
            delta = max(delta, abs(new_v[s] - v[s]))
        v = new_v  # only update v after iteration ends
        v_history.append(v.copy())
        policy_history.append(policy)

        # break if converged
        print(f"converged? delta = {delta:.6f}")
        if delta <= convergence_threshold:
            break

        k += 1
    return k, policy_history, q, v_history


@app.cell
def _(k):
    k
    return


@app.cell
def _(q):
    q
    return


@app.cell
def _(np, policy_history: "list[np.ndarray]"):
    np.array(policy_history, dtype=np.float64)
    return


@app.cell
def _(np, v_history: "list[np.ndarray]"):
    np.array(v_history, dtype=np.float64)
    return


if __name__ == "__main__":
    app.run()

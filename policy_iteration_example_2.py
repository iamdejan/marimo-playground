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
    ## Example 2
    """)
    return


@app.cell
def _(mo):
    mo.image(src="./images/policy_iteration_example_2.png")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Initialize environment
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
        Tuple,
        actions,
        calculate_reward,
        convergence_threshold,
        discount_rate,
        is_out_of_bounds,
        jaxtyped,
        np,
        reward_probability,
        size,
        state_transition_probability,
        typechecker,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Search for optimal value and optimal policy
    """)
    return


@app.cell
def _(
    Tuple,
    actions: "list[Tuple[int, int]]",
    calculate_reward,
    convergence_threshold: float,
    discount_rate: float,
    is_out_of_bounds,
    jaxtyped,
    np,
    reward_probability: float,
    size,
    state_transition_probability: float,
    typechecker,
):
    import math

    from jaxtyping import Float64

    # Initialize value state
    v = np.zeros(shape=(size[0] * size[1],), dtype=np.float64)
    state_space_size = len(v)
    v_history: list[np.ndarray] = []

    # Initialize q-table
    actions_len = len(actions)
    q = np.zeros(shape=(state_space_size, actions_len), dtype=np.float64)
    q_history: list[np.ndarray] = []

    # Initialize policy
    policy = np.zeros(shape=(actions_len, state_space_size), dtype=np.float64)
    for s in range(state_space_size):
        policy[4, s] = 1
    print("Initial policy:\n", policy)
    policy_stable = False


    @jaxtyped(typechecker=typechecker)
    def policy_evaluation_one_iteration(
        state_space_size: int,
        policy: Float64[np.ndarray, "actions_len state_space_size"],
        actions: list[Tuple[int, int]],
        v: Float64[np.ndarray, "state_space_size"],
        j: int,
    ) -> Tuple[Float64[np.ndarray, "state_space_size"], float]:
        new_v = np.zeros_like(v, dtype=np.float64)
        delta: float = 0.0
        for s in range(state_space_size):
            r = math.floor(s / size[0])
            c = s % size[1]

            a = np.argmax(policy[:, s])
            move: Tuple[int, int] = actions[a]
            next_r: int = r + move[0]
            next_c: int = c + move[1]
            immediate_reward = calculate_reward(next_r, next_c)
            if is_out_of_bounds(next_r, next_c):
                # bounce back
                next_r = r
                next_c = c
            next_s = math.floor(next_r * size[0] + next_c)
            v_next_state: float = v[next_s]
            new_v[s] = policy[a, s] * (
                reward_probability * immediate_reward
                + discount_rate * state_transition_probability * v_next_state
            )
            print(
                f"[STEP 1] v_{k}^{j}[{s}] = {policy[a, s]} * ({immediate_reward} + {discount_rate} * {v_next_state}) = {new_v[s]}"
            )
            delta = max(delta, abs(new_v[s] - v[s]))

        return new_v, delta


    @jaxtyped(typechecker=typechecker)
    def policy_evaluation(
        state_space_size: int,
        policy: Float64[np.ndarray, "actions_len state_space_size"],
        actions: list[Tuple[int, int]],
        v: Float64[np.ndarray, "state_space_size"],
    ) -> Float64[np.ndarray, "state_space_size"]:
        j = 1
        while True:
            new_v, delta = policy_evaluation_one_iteration(
                state_space_size,
                policy,
                actions,
                v,
                j,
            )

            # without .copy(), v will have the same reference as new_v,
            # thus messing with the values
            v = new_v.copy()
            j += 1

            # break if converged
            if delta <= convergence_threshold:
                break

        return v


    @jaxtyped(typechecker=typechecker)
    def policy_improvement(
        actions: list[Tuple[int, int]],
        state_space_size: int,
        q: Float64[np.ndarray, "state_space_size actions_len"],
    ) -> Tuple[
        Float64[np.ndarray, "actions_len state_space_size"],
        Float64[np.ndarray, "state_space_size actions_len"],
        bool,
    ]:
        new_policy = np.zeros_like(policy, dtype=np.float64)
        policy_stable = True
        for s in range(state_space_size):
            r = math.floor(s / size[0])
            c = s % size[1]

            best_a = None
            max_q = -float("inf")
            for a in range(len(actions)):
                move: Tuple[int, int] = actions[a]
                next_r: int = r + move[0]
                next_c: int = c + move[1]
                immediate_reward = calculate_reward(next_r, next_c)
                if is_out_of_bounds(next_r, next_c):
                    next_r = r
                    next_c = c
                next_s = next_r * size[0] + next_c
                v_next_state: float = v[next_s]
                q[s, a] = (
                    reward_probability * immediate_reward
                    + discount_rate * state_transition_probability * v_next_state
                )

                if q[s, a] > max_q:
                    max_q = q[s, a]
                    best_a = a

            new_policy[best_a, s] = 1
            current_action = np.argmax(policy[:, s])
            if current_action != best_a:
                policy_stable = False

        return new_policy, q, policy_stable


    q_history.append(q.copy())
    k = 1
    while True:
        # Step 1: policy evaluation
        v = policy_evaluation(state_space_size, policy, actions, v)
        v_history.append(v.copy())

        # Step 2: policy improvement
        new_policy, q, policy_stable = policy_improvement(actions, state_space_size, q)
        q_history.append(q.copy())

        policy = new_policy
        k += 1

        # break if policy is stable
        if policy_stable:
            break
    return k, policy, v_history


@app.cell
def _(k):
    k
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
def _(np, v_history: "list[np.ndarray]"):
    def print_data(data):
        i = 0
        while i < len(data):
            for w in range(5):
                print(data[i], end=" \t")
                i += 1
            print("")


    for v_data in v_history:
        print_data(np.round(v_data, decimals=1))
        print("")
    return


if __name__ == "__main__":
    app.run()

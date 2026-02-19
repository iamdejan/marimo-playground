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
    # Policy Iteration

    Policy iteration is another algorithm to find the optimal policy. Unlike value iteration, this algorithm doesn't directly solve Bellman optimality equation. However, it has an intimate relationship with value iteration, as shown later. Moreover, the idea of policy iteration is very important since it is widely utilized in reinforcement learning algorithms.

    Policy iteration is an iterative algorithm. Each iteration has two steps:
    1. Policy evaluation step: this step evaluates a given policy by calculating the corresponding state value.
    2. Policy improvement step: this step is used to improve the policy. In particular, once $v_{\pi_{k}}$ has been calculated in the first step, a new policy $\pi_{k+1}$ can be obtained as: $\pi_{k+1} = \text{arg} \underset{\pi}{\text{max}}(r_{\pi} + \gamma P_{\pi} v_{\pi_{k}})$

    Three questions naturally follow:
    - How to solve the state value $v_{\pi_{k}}$?
    - In the policy improvement step, why is the new policy $\pi_{k+1}$ better than $\pi_{k}$?
    - Why can this algorithm converge to optimal policy?

    ## How to solve the state value $v_{\pi_{k}}$?

    There are 2 ways:
    - Closed form: $v_{\pi} = (I - \gamma P_{\pi})^{-1} r_{\pi}$
    - Iterative algorithm: $v_{\pi_{k}}^{(j+1)} = r_{\pi_{k}} + \gamma P_{\pi_{k}} v_{\pi_{k}}^{(j)}$. See [here](https://share.google/aimode/FEmRNlUZ0OGEvYjn7) for the difference between $j$ and $k$.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Example 1
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.image(src="./images/policy_iteration_example_1.png")
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

    size = (1, 2)  # (row, column)
    actions = [
        (0, -1),  # move to the left
        (0, 0),  # stay
        (0, 1),  # move to the right
    ]

    reward_probability = 1.0  # p(r|s,a)
    state_transition_probability = 1.0  # p(s'|s,a)
    convergence_threshold = 1e-4
    discount_rate = 0.9

    # Initialize reward
    reward_boundary = -1
    reward_goal = 1
    goal = (0, 1)


    def is_out_of_bounds(c: int) -> bool:
        return c < 0 or c >= size[1]


    def calculate_reward(c: int) -> int:
        if is_out_of_bounds(c):
            return reward_boundary

        if (0, c) == goal:
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
    ### Search for optimal value and optimal policy
    """)
    return


@app.cell
def _(
    actions,
    calculate_reward,
    convergence_threshold,
    discount_rate,
    is_out_of_bounds,
    np,
    reward_probability,
    size,
    state_transition_probability,
):
    from typing import Tuple

    # Initialize value state
    v = np.zeros(shape=(size[0] * size[1],), dtype=np.float64)
    state_space_size = len(v)

    # Initialize q-table
    q = np.zeros(shape=(state_space_size, len(actions)), dtype=np.float64)

    # Initialize policy
    policy = np.zeros(shape=(len(actions), state_space_size), dtype=np.float64)
    policy[actions[0], 0] = 1
    policy[actions[0], 1] = 1
    policy_stable = False


    def policy_evaluation(
        state_space_size: int, policy: np.ndarray, actions: list[Tuple[int, int]], v: np.ndarray
    ) -> np.ndarray:
        new_v = np.zeros_like(v)

        j = 1
        while True:
            delta = 0
            for s in range(state_space_size):
                c = s % size[1]

                a = np.argmax(policy[:, s])
                move = actions[a]
                next_c = c + move[1]
                immediate_reward = calculate_reward(next_c)
                if is_out_of_bounds(next_c):
                    # bounce back
                    next_c = c
                v_next_state = v[next_c]
                new_v[s] = policy[a, s] * (
                    reward_probability * immediate_reward
                    + discount_rate * state_transition_probability * v_next_state
                )
                print(
                    f"[STEP 1] v_{k}^{j}[{s}] = {policy[a, s]} * ({immediate_reward} + {discount_rate} * {v_next_state}) = {new_v[s]}"
                )
                delta = max(delta, abs(new_v[s] - v[s]))

            # without .copy(), v will have the same reference as new_v,
            # thus messing with the values
            v = new_v.copy()
            j += 1

            # break if converged
            if delta <= convergence_threshold:
                break

        return v


    def policy_improvement(
        actions: list[Tuple[int, int]], state_space_size: int, q: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, bool]:
        new_policy = np.zeros(shape=(len(actions), state_space_size), dtype=np.float64)
        policy_stable = True
        for s in range(state_space_size):
            c = s % size[1]

            best_a = None
            max_q = -float("inf")
            for a in range(len(actions)):
                next_c = c + actions[a][1]
                immediate_reward = calculate_reward(next_c)
                if is_out_of_bounds(next_c):
                    next_c = c
                v_next_state = v[next_c]
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


    k = 1
    while True:
        # Step 1: policy evaluation
        v = policy_evaluation(state_space_size, policy, actions, v)

        # Step 2: policy improvement
        new_policy, q, policy_stable = policy_improvement(actions, state_space_size, q)

        policy = new_policy
        k += 1

        # break if policy is stable
        if policy_stable:
            break
    return k, policy


@app.cell
def _(k):
    k
    return


@app.cell
def _(policy):
    policy
    return


if __name__ == "__main__":
    app.run()

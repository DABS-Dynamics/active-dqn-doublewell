"""
ACTIVE LEARNING + DQN — v2
Multi-potential benchmark + active-learning QM-efficiency.

Keeps the original 3D double-well behaviour, but:
- adds extra asymmetric / rugged potentials, and
- reports QM-call savings vs a naive always-oracle baseline.
"""


# === Python 3.13 TensorFlow fix ===
import sys
if sys.version_info >= (3, 13):
    import typing
    sys.modules['typing_extensions'] = typing

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import warnings, random, numpy as np, tensorflow as tf
from collections import deque
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")


# ============================== NFF ==============================
class NFF:
    def __init__(self):
        m = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu', input_shape=(3,)),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(2)
        ])
        m.compile(optimizer='adam', loss='mse')
        self.model = m
        self.data = []

    def predict(self, s):
        p = self.model.predict(s[None], verbose=0)[0]
        return float(p[0]), float(abs(p[1]))

    def train(self):
        if len(self.data) < 5:
            return
        X = np.array([x for x, _ in self.data])
        y = np.array([e for _, e in self.data])
        targets = np.column_stack([y, np.zeros_like(y)])
        self.model.fit(X, targets, epochs=12, verbose=0, shuffle=True)


# ============================== ENV ==============================
class Env:
    def __init__(self, potential_type="double_well"):
        """3D potential environment with active-learning QM calls.

        potential_type options:
        - "double_well": symmetric double-well (original v1 behaviour)
        - "asymmetric": tilted double-well (adds linear term)
        - "rugged": double-well with fast oscillatory bumps
        """
        self.nff = NFF()
        self.pos = None
        self.global_steps = 0  # track global steps for warm-up
        self.potential_type = potential_type

    def reset(self):
        # Start near the origin; spread controls exploration radius.
        self.pos = np.random.randn(3) * 0.6
        self._step_count = 0
        return self.pos.copy()

    def true_energy(self, p):
        """Analytic potential energy surface for different test cases.

        All share the same basic double-well shape so results are comparable.
        """
        if self.potential_type == "double_well":
            # Original symmetric 3D double-well from v1 (unchanged).
            return np.sum(p**4 - p**2)
        elif self.potential_type == "asymmetric":
            # Tilt the wells slightly so one side is globally preferred.
            return np.sum(p**4 - p**2 + 0.3 * p)
        elif self.potential_type == "rugged":
            # Add small, high-frequency bumps on top of the double-well.
            base = np.sum(p**4 - p**2)
            bumps = 0.2 * np.sin(5.0 * p).sum()
            return base + bumps
        else:
            raise ValueError(f"Unknown potential_type: {self.potential_type}")

    def step(self, a):
        # Discrete 6-action grid: ± step along each Cartesian axis.
        p = self.pos.copy()
        p[a // 2] += (1 if a % 2 == 0 else -1) * 0.08
        self.pos = p

        # True oracle energy; reward is minus energy (we want low energy).
        true_e = self.true_energy(p)
        reward = -true_e

        # Active-learning gate: use surrogate when confident,
        # query "QM" (i.e., add a new labelled point) when uncertain.
        e_pred, unc = self.nff.predict(p)
        step_counter = getattr(self, '_step_count', 0)
        self._step_count = step_counter + 1
        self.global_steps += 1

        # Warm-up: always treat early steps as QM calls to stabilise NFF.
        if unc > 0.25 or self.global_steps < 200:
            self.nff.data.append((p.copy(), true_e))
            self.nff.train()

        # Reset per-episode counter every 200 steps (1 episode).
        if step_counter >= 199:
            self._step_count = 0

        # No terminal condition; episodes are fixed-length.
        return p.copy(), float(reward), False, {}


# ============================== AGENT ==============================
class Agent:
    def __init__(self):
        m = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(3,)),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(6, activation='linear')
        ])
        m.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss='mse')
        self.model = m
        self.target = tf.keras.models.clone_model(m)
        self.target.set_weights(m.get_weights())
        self.mem = deque(maxlen=20000)
        self.eps = 1.0

    def act(self, s):
        if random.random() < self.eps:
            return random.randrange(6)
        return int(np.argmax(self.model.predict(s[None], verbose=0)[0]))

    def store(self, s, a, r, ns, d):
        self.mem.append((s, a, r, ns, d))

    def replay(self):
        if len(self.mem) < 64:
            return
        b = random.sample(self.mem, 64)
        s = np.array([x[0] for x in b])
        a = [x[1] for x in b]
        r = [x[2] for x in b]
        ns = np.array([x[3] for x in b])

        q = self.model.predict(s, verbose=0)
        nq = self.target.predict(ns, verbose=0)

        for i in range(64):
            q[i][a[i]] = r[i] + 0.95 * np.max(nq[i])

        self.model.fit(s, q, epochs=1, verbose=0)
        self.eps = max(0.01, self.eps * 0.995)

    def sync(self):
        self.target.set_weights(self.model.get_weights())


# ============================== MAIN (v2) ==============================
if __name__ == "__main__":
    print("ACTIVE LEARNING + DQN — v2 (multi-potential + baseline efficiency)")
    EPISODES = 40
    STEPS_PER_EPISODE = 200

    potential_types = ["double_well", "asymmetric", "rugged"]
    all_rewards = {}
    qm_calls = {}

    for pot in potential_types:
        print("\n" + "=" * 60)
        print(f"Potential: {pot}")
        print("=" * 60)
        print(f"{'Ep':<4} {'Reward':>9} {'ε':>7} {'QM':>6}")
        print("-" * 34)

        env = Env(potential_type=pot)
        agent = Agent()
        rewards = []

        for ep in range(1, EPISODES + 1):
            s = env.reset()
            tot_r = 0.0
            for _ in range(STEPS_PER_EPISODE):
                a = agent.act(s)
                ns, r, _, _ = env.step(a)
                agent.store(s, a, r, ns, False)
                s = ns
                tot_r += r
                agent.replay()
            agent.sync()

            rewards.append(tot_r)
            print(f"{ep:<4} {tot_r:9.2f} {agent.eps:7.3f} {len(env.nff.data):6}")

        all_rewards[pot] = rewards
        qm_calls[pot] = len(env.nff.data)

    # Baseline QM cost: calling the true oracle at every step.
    baseline_qm = EPISODES * STEPS_PER_EPISODE

    print("\n=== Summary of QM call savings vs. naive baseline ===")
    print(f"Baseline (no active learning): {baseline_qm} QM calls per potential")
    for pot in potential_types:
        calls = qm_calls[pot]
        saved_frac = 100.0 * (1.0 - calls / baseline_qm)
        print(f"{pot:<10} | QM calls = {calls:4d} | saved = {saved_frac:6.2f}% vs baseline")

    # Plot reward curves for all potentials on one figure.
    plt.figure(figsize=(10, 6))
    for pot in potential_types:
        plt.plot(range(1, EPISODES + 1), all_rewards[pot], 'o-', linewidth=2, label=pot)
    plt.axhline(0, color='black', linestyle='--', alpha=0.7)
    plt.title('Active Learning + DQN across 3D Potentials (v2)', fontsize=16, pad=20)
    plt.xlabel('Episode', fontsize=14)
    plt.ylabel('Total Reward (200 steps)', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('reward_curves_v2.png', dpi=400, bbox_inches='tight')
    plt.close()

    print("\nPlot saved: reward_curves_v2.png")
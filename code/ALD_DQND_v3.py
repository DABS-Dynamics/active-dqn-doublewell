"""
ACTIVE LEARNING + DQN — v3
Ensemble-based uncertainty + OOD guard for asymmetric failures.

- Uses an ensemble of NFF models (K small networks) as the surrogate.
- Uncertainty = std-dev of ensemble predictions.
- Active learning triggers QM calls when:
    - ensemble disagreement is high, OR
    - the agent goes too far from the training region (OOD radius guard).
"""

# === Python 3.13 TensorFlow fix ===
import sys
if sys.version_info >= (3, 13):
    import typing
    sys.modules['typing_extensions'] = typing

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import warnings
warnings.filterwarnings("ignore")

import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from collections import deque


# ============================== CONFIG ==============================
EPISODES = 40
STEPS_PER_EPISODE = 200

# Surrogate ensemble settings
N_ENSEMBLE = 5
NFF_MIN_POINTS = 10          # minimum data before training
NFF_EPOCHS_PER_UPDATE = 8    # per-model epochs on each AL update
NFF_BATCH_SIZE = 64          # batch size for NFF training (if enough data)

# Active learning thresholds
WARMUP_STEPS = 200           # always query QM for the first 200 global steps
UNC_THRESHOLD = 0.3          # ensemble std dev threshold for "I don't know"

# OOD / domain guard
POS_CLIP = 4.0               # hard clamp for positions (keeps things numerically sane)
OOD_RADIUS = 2.8             # beyond this norm, treat as out-of-distribution
OOD_UNC_BOOST = 10.0         # added to uncertainty when OOD


# ============================== NFF ENSEMBLE ==============================
def build_nff_model():
    """Build a small MLP surrogate: R^3 -> R (energy)."""
    m = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(3,)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1, activation='linear'),
    ])
    m.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss='mse')
    return m


class NFFEnsemble:
    """Ensemble surrogate for PES with disagreement-based uncertainty."""

    def __init__(self, n_members=N_ENSEMBLE):
        self.models = [build_nff_model() for _ in range(n_members)]
        self.data = []  # list of (x, E_true)

    def predict(self, p):
        """
        Returns:
            mean_energy (float)
            unc (float): ensemble std dev, boosted if clearly OOD.
        """
        p = np.asarray(p, dtype=np.float32).reshape(1, -1)
        preds = []
        for m in self.models:
            e = m.predict(p, verbose=0)[0, 0]
            preds.append(float(e))
        preds = np.array(preds)
        mean_e = float(preds.mean())
        std_e = float(preds.std())

        # Distance-based OOD guard: if we're far away from origin, uncertainty must be high.
        norm_p = float(np.linalg.norm(p))
        if norm_p > OOD_RADIUS:
            std_e += OOD_UNC_BOOST

        return mean_e, std_e

    def maybe_train(self):
        """Train all ensemble members if enough data exists."""
        if len(self.data) < NFF_MIN_POINTS:
            return

        X = np.array([x for x, _ in self.data], dtype=np.float32)
        y = np.array([e for _, e in self.data], dtype=np.float32).reshape(-1, 1)

        # If small dataset, just train on full batch.
        if len(self.data) <= NFF_BATCH_SIZE:
            X_batch = X
            y_batch = y
        else:
            idx = np.random.choice(len(self.data), size=NFF_BATCH_SIZE, replace=True)
            X_batch = X[idx]
            y_batch = y[idx]

        for m in self.models:
            m.fit(X_batch, y_batch,
                  epochs=NFF_EPOCHS_PER_UPDATE,
                  verbose=0,
                  shuffle=True)


# ============================== ENVIRONMENT ==============================
class Env:
    """
    3D potential environment with ensemble surrogate and active-learning QM calls.

    potential_type options:
      - "double_well"
      - "asymmetric"
      - "rugged"
    """

    def __init__(self, potential_type="double_well"):
        self.nff = NFFEnsemble()
        self.pos = None
        self.global_steps = 0
        self.potential_type = potential_type
        self._step_count = 0

    def reset(self):
        # Start near the origin with modest spread.
        self.pos = np.random.randn(3).astype(np.float32) * 0.6
        self._step_count = 0
        return self.pos.copy()

    def true_energy(self, p):
        """Analytic potentials used as QM oracle."""
        p = np.asarray(p, dtype=np.float32)
        if self.potential_type == "double_well":
            # Symmetric 3D double-well.
            return float(np.sum(p**4 - p**2))
        elif self.potential_type == "asymmetric":
            # Tilted double-well: one side energetically preferred, other side goes to hell.
            return float(np.sum(p**4 - p**2 + 0.3 * p))
        elif self.potential_type == "rugged":
            # Double-well with sinusoidal bumps.
            base = np.sum(p**4 - p**2)
            bumps = 0.2 * np.sin(5.0 * p).sum()
            return float(base + bumps)
        else:
            raise ValueError(f"Unknown potential_type: {self.potential_type}")

    def step(self, a):
        """
        Discrete 6-action grid: ± step along each Cartesian axis.
        a ∈ {0,...,5}: axis = a // 2, sign = +1 if even, -1 if odd.
        """
        p = self.pos.copy()
        axis = a // 2
        step_dir = 1.0 if (a % 2 == 0) else -1.0
        p[axis] += step_dir * 0.08

        # Hard clamp to keep things numerically controlled.
        p = np.clip(p, -POS_CLIP, POS_CLIP)
        self.pos = p

        # True QM energy and RL reward.
        true_e = self.true_energy(p)
        reward = -true_e  # RL wants to minimize energy.

        # Ensemble surrogate prediction + uncertainty.
        mean_e, unc = self.nff.predict(p)

        # Global step counting.
        self._step_count += 1
        self.global_steps += 1

        # Active-learning gate:
        #   - Always collect data during warmup.
        #   - After warmup, query QM only if ensemble disagreement is high.
        if (self.global_steps <= WARMUP_STEPS) or (unc > UNC_THRESHOLD):
            self.nff.data.append((p.copy(), true_e))
            self.nff.maybe_train()

        # Fixed-length episodes: no terminal state flag used by agent.
        if self._step_count >= STEPS_PER_EPISODE:
            self._step_count = 0

        return p.copy(), float(reward), False, {}


# ============================== AGENT (DQN) ==============================
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
        q = self.model.predict(s[None], verbose=0)[0]
        return int(np.argmax(q))

    def store(self, s, a, r, ns, d):
        self.mem.append((s, a, r, ns, d))

    def replay(self):
        if len(self.mem) < 64:
            return
        batch = random.sample(self.mem, 64)
        s = np.array([x[0] for x in batch], dtype=np.float32)
        a = [x[1] for x in batch]
        r = [x[2] for x in batch]
        ns = np.array([x[3] for x in batch], dtype=np.float32)

        q = self.model.predict(s, verbose=0)
        nq = self.target.predict(ns, verbose=0)

        for i in range(64):
            q[i][a[i]] = r[i] + 0.95 * float(np.max(nq[i]))

        self.model.fit(s, q, epochs=1, verbose=0)
        self.eps = max(0.01, self.eps * 0.995)

    def sync(self):
        self.target.set_weights(self.model.get_weights())


# ============================== MAIN (v3) ==============================
if __name__ == "__main__":
    print("ACTIVE LEARNING + DQN — v3 (ensemble uncertainty + OOD guard)")
    potential_types = ["double_well", "asymmetric", "rugged"]
    all_rewards = {}
    qm_calls = {}

    for pot in potential_types:
        print("\n" + "=" * 60)
        print(f"Potential: {pot}")
        print("=" * 60)
        print(f"{'Ep':<4} {'Reward':>10} {'ε':>7} {'QM':>6}")
        print("-" * 40)

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
            print(f"{ep:<4} {tot_r:10.2f} {agent.eps:7.3f} {len(env.nff.data):6}")

        all_rewards[pot] = rewards
        qm_calls[pot] = len(env.nff.data)

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
        plt.plot(
            range(1, EPISODES + 1),
            all_rewards[pot],
            'o-',
            linewidth=2,
            label=pot
        )
    plt.axhline(0, color='black', linestyle='--', alpha=0.7)
    plt.title('Active Learning + DQN with Ensemble Surrogate (v3)', fontsize=16, pad=20)
    plt.xlabel('Episode', fontsize=14)
    plt.ylabel('Total Reward (200 steps)', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('reward_curves_v3.png', dpi=400, bbox_inches='tight')
    plt.close()

    print("\nPlot saved: reward_curves_v3.png")

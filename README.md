# **Active-Learning-Driven Deep Q-Network (Active DQN)**

**Discovers the Global Minimum of a 3D Double-Well Potential with Extreme Data Efficiency**

## **📌 Overview**

This repository contains the implementation of a discrete-action **Deep Q-Network (DQN)** guided by an uncertainty-aware **Neural Force Field (NFF)**.

The agent successfully navigates a rugged, non-convex **3D Isotropic Double-Well Potential** ($E(\\mathbf{r}) \= \\sum (r\_i^4 \- r\_i^2)$) to find the global minimum. By utilizing an Active Learning scheme, the system requests high-fidelity "Quantum Mechanical" (QM) evaluations **only when uncertainty is high**.

**Key Result:** The agent learns the physics within the first 200 steps (1 episode) and solves the optimization problem for the remaining 39 episodes with **zero additional oracle calls**.

## **🚀 Key Features**

* **Discrete Action Space:** The agent moves in small discrete steps ($\\pm 0.08$) along Cartesian axes, avoiding the complexity of continuous control.  
* **Uncertainty-Driven Active Learning:** A neural surrogate model estimates its own epistemic uncertainty. The expensive "ground truth" function is only called when this uncertainty exceeds a threshold ($u \> 0.25$).  
* **Non-Convex Optimization:** Solves a "W"-shaped potential with local traps and high-energy barriers, proving robustness beyond simple harmonic oscillators.  
* **Extreme Efficiency:** Solves the environment using only **199 exact evaluations** out of 8,000 total steps (97.5% data efficiency).

## **🛠️ Installation**

1. Clone the repository:  
   git clone \[https://github.com/StonerIsh420/active-dqn-doublewell.git\](https://github.com/StonerIsh420/active-dqn-doublewell.git)  
   cd active-dqn-doublewell

2. Install dependencies (TensorFlow, NumPy, Matplotlib):  
   pip install tensorflow numpy matplotlib

   *Note: The code includes a compatibility fix for Python 3.13+.*

## **💻 Usage**

Run the main training script. This will train the DQN, perform active learning, and generate the reward plot.

python ALD\_DQND.py

**Output:**

* Real-time logs of Episode Reward, Epsilon, and Total QM Calls.  
* A generated plot file: ![Output Plot](plot/reward_curve_example.png).

## **🔬 The Physics**

The environment simulates a particle in a 3D Isotropic Double-Well potential. Unlike a simple bowl (Harmonic Oscillator), this potential has local maxima (barriers) and multiple minima.

The Equation:

$$E(\\mathbf{r}) \= \\sum\_{i \\in \\{x,y,z\\}} (r\_i^4 \- r\_i^2)$$

* **Global Minima:** Located at $r\_i \\approx \\pm 0.707$ for each axis.  
* **Global Minimum Energy:** $-0.75$  
* **Max Possible Reward:** $150.0$ (over 200 steps).

## **📊 Results Summary**

| Metric | Value | Note |
| :---- | :---- | :---- |
| **Total Episodes** | 40 | 200 steps each |
| **Total QM Calls** | **199** | **Plateaus after Episode 1** |
| **Convergence Reward** | \~129 | \~85% of Theoretical Max |
| **Success Rate** | 100% | Found global minimum in all late episodes |

For a detailed breakdown, refer to the output log file. 
![Run Ouput](run_output/run_results.sh)

## **📄 Citation**

If you use this code or methodology in your research, please cite the accompanying paper:

@article{Stoner2025ActiveDQN,  
  title={Active-Learning-Driven Deep Q-Network Discovers the Global Minimum of a 3D Double-Well Potential Using Only Discrete Actions and 199 High-Fidelity Evaluations},  
  author={Stoner},  
  journal={Independent Research},  
  year={2025},  
  month={December}  
}

## **🙌 Acknowledgments**

* **Grok (xAI) & Gemini (Google):** For extensive assistance in code debugging, manuscript refinement, and ensuring strict alignment between the implementation and the text.  
* **Original Concept:** The conceptual framework and research design are the sole work of the author.

## **📜 License**

This project is licensed under the MIT License \- see the LICENSE file for details.
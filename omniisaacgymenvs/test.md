# Paper 1 - Release Notes

## Overview

This release corresponds to the version of the code used in the research paper titled "Methods for Controlling Floating Platforms in 2D Space". The problem formulation, simulation details, training procedure, and benchmarks discussed in the paper are based on this version.

## Problem Formulation

The problem is formulated as a sequential decision-making task to control a floating platform's maneuvers within a 2D space. The state space, actions, and task-specific observations are defined as per the equations and tables provided in the paper.

## Reward Functions

Three reward functions for different tasks (Go to position, Go to pose, Track velocity) are defined as exponential terms in the paper. These reward functions have been utilized for training the agents in this version.

## Simulation

The simulation enhancements based on the RANS framework (RANS v2.0) have been integrated to perform more complex tasks. It includes parameterized rewards, penalties, disturbance generators, and allows action and state noises to be injected. The simulator achieves a high throughput of more than 40,000 steps per second.

## Training Procedure

The training procedure is based on the PPO (Proximal Policy Optimization) algorithm with specific network configurations and training details provided in the paper. The agents undergo training for a total of 2000 epochs or approximately 130M steps.

## Benchmark Comparison

This version includes a benchmark comparison between deep reinforcement learning (DRL) and optimal control approaches (LQR) for controlling the floating platform. The comparison aims to provide insights into the strengths and weaknesses of each approach.

## Optimal Controller

An infinite horizon discrete-time LQR controller is implemented to compare with the DRL algorithm for controlling the floating platform. The state variables and corresponding state matrices for the LQR controller are calculated using finite differencing.

## Laboratory Experiment Setup

Real-world validation experiments were conducted using the physical air bearings platform located in the ZeroG Laboratory at the University of Luxembourg. Details about the laboratory setup and experimental procedures can be found in the paper.

---

For the most up-to-date information and code, please refer to the [main repository](https://github.com/yourusername/yourrepository).

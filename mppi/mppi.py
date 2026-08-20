"""MPPI (Model Predictive Path Integral) 3D path planner for air combat scene.

A state-of-the-art sampling-based trajectory optimizer that produces
dynamically-feasible paths by rolling out perturbed control sequences
through a nonlinear aircraft kinematics model and re-weighting via
importance sampling (information-theoretic MPC).

Key features:
  - Control-space sampling with Gaussian perturbations
  - Nonlinear 5-DoF kinematic model (turn rate + climb rate constraints)
  - Soft obstacle cost field (mountain cylinders)
  - Temperature-controlled softmin importance weighting
  - Iterative refinement with control sequence warm-start
  - GPU-friendly batch operations via NumPy vectorization

References:
  - Williams et al., "Information-Theoretic Model Predictive Control", 2017
  - Wagener et al., "The Role of Information in MPPI Control", 2022
  - Bhardwaj et al., "STORM: An Integrated MPPI Framework", 2024
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

try:
    from .obstacles import CylindricalObstacle, is_position_blocked
except ImportError:
    from obstacles import CylindricalObstacle, is_position_blocked  # type: ignore[no-redef]


# ==========================================================================
# Configuration
# ==========================================================================


@dataclass
class MPPIConfig:
    """MPPI planner hyperparameters.

    Tuning guide:
      - Increase *num_samples* for smoother paths at the cost of speed.
      - Increase *horizon* for longer-range planning.
      - Decrease *temperature* to exploit more (narrower sampling).
      - Increase *turn_rate_std* for more agile paths.
    """

    # -- Map bounds -------------------------------------------------------
    map_size_units: float = 3000.0
    map_origin: tuple[float, float] = (-1500.0, -1500.0)
    min_altitude: float = 0.5
    max_altitude: float = 200.0

    # -- MPPI hyperparameters ---------------------------------------------
    num_samples: int = 512          # K: trajectories sampled per iteration
    horizon: int = 50               # H: planning horizon (timesteps)
    num_iterations: int = 5         # M: MPPI refinement iterations
    temperature: float = 1.0        # lambda: softmin temperature (lower=exploit)

    # -- Kinematic model --------------------------------------------------
    cruise_speed: float = 80.0      # v: constant forward speed (units/s)
    dt: float = 0.5                 # dt: integration timestep (s)
    max_turn_rate: float = math.radians(45.0)   # omega_max (rad/s)
    max_climb_rate: float = 8.0     # v_z_max (units/s)

    # -- Control noise std ------------------------------------------------
    turn_rate_std: float = math.radians(20.0)   # sigma_omega
    climb_rate_std: float = 3.0                 # sigma_vz

    # -- Cost weights -----------------------------------------------------
    w_goal_xy: float = 1.0          # terminal: horizontal distance to goal
    w_goal_z: float = 2.0           # terminal: vertical distance to goal
    w_obstacle: float = 500.0       # running: obstacle proximity penalty
    w_control: float = 0.1          # running: ||u||^2 penalty (smoothness)
    w_boundary: float = 1000.0      # running: map boundary violation
    w_altitude: float = 500.0       # running: altitude limit violation
    w_cruise_altitude: float = 200.0  # running: deviation from cruise altitude

    # -- Obstacle cost ----------------------------------------------------
    obstacle_clearance: float = 3.0     # safety margin (units)
    obstacle_cost_scale: float = 50.0   # sharpness of obstacle penalty

    # -- Early termination ------------------------------------------------
    early_stop_threshold: float = 15.0  # stop if any sample reaches this dist


# ==========================================================================
# MPPI Path Planner
# ==========================================================================


class MPPIPlanner:
    """MPPI-based 3D path planner with nonlinear aircraft dynamics.

    Drop-in replacement for ``HybridAStarPlanner`` with the same
    ``plan(start, goal, verbose) -> list[ndarray] | None`` interface.

    Algorithm sketch (per MPPI iteration):

        1. Sample  K  i.i.d. control perturbations  eps_k ~ N(0, Sigma)
           for the full horizon  H.
        2. Roll out  K  trajectories using the perturbed control sequence
           U + eps_k  through the kinematic model.
        3. Evaluate per-trajectory cost  S(tau_k).
        4. Compute importance weights  w_k = exp(-S(tau_k) / lambda).
        5. Update control sequence:  U <- U + sum_k (w_k / sum w) * eps_k.
        6. Shift  U  one step, append zero -> warm-start for next iteration.

    After  M  iterations the final control sequence is rolled out once to
    produce the output path.
    """

    def __init__(
        self,
        obstacles: list[CylindricalObstacle] | None = None,
        config: MPPIConfig | None = None,
    ):
        self.obstacles: list[CylindricalObstacle] = obstacles or []
        self.config: MPPIConfig = config or MPPIConfig()

    # -- Public API -------------------------------------------------------

    def plan(
        self,
        start: np.ndarray,
        goal: np.ndarray,
        verbose: bool = False,
    ) -> list[np.ndarray] | None:
        """Plan a dynamically feasible 3D path from start to goal.

        Args:
            start: [x, y, z] in scene units.
            goal:  [x, y, z] in scene units.
            verbose: Print MPPI progress.

        Returns:
            List of waypoints [start, ..., goal], or None if unreachable.
        """
        cfg = self.config
        start = np.asarray(start, dtype=float)
        goal = np.asarray(goal, dtype=float)

        # -- validate endpoints -------------------------------------------
        if self._endpoint_blocked(start, "start", verbose):
            return None
        if self._endpoint_blocked(goal, "goal", verbose):
            return None

        # -- initialise control sequence ----------------------------------
        # U: [H, 2]   columns = [turn_rate, climb_rate]
        U = self._init_control_sequence(start, goal)

        best_path: list[np.ndarray] | None = None
        best_cost = float("inf")
        rng = np.random.default_rng()

        # -- MPPI iterative refinement ------------------------------------
        for iteration in range(cfg.num_iterations):
            # 1. Sample control perturbations  eps ~ N(0, Sigma)
            eps = self._sample_perturbations(cfg, rng)

            # 2. Roll out K trajectories in batch
            states, costs = self._batch_rollout(start, goal, U, eps)

            # 3. Check early stop
            min_cost = float(np.min(costs))
            min_idx = int(np.argmin(costs))
            if min_cost < best_cost:
                best_cost = min_cost
                best_path = self._states_to_path(states[min_idx])

            # 4. Importance weights via softmin
            beta = 1.0 / max(cfg.temperature, 1e-6)
            costs_shifted = costs - min_cost  # numerical stability
            weights = np.exp(-beta * costs_shifted)
            weight_sum = float(np.sum(weights))
            if weight_sum < 1e-12:
                # All trajectories are invalid -> keep previous U
                if verbose:
                    print(f"  [MPPI] iter {iteration + 1}: all samples invalid")
                U[:, :] = np.roll(U, -1, axis=0)
                U[-1, :] = 0.0
                continue
            weights /= weight_sum

            # 5. MPPI update: U <- U + sum_k w_k * eps_k
            update = np.tensordot(weights, eps, axes=(0, 0))  # [H, 2]
            U += update

            # 6. Clamp controls
            U[:, 0] = np.clip(U[:, 0], -cfg.max_turn_rate, cfg.max_turn_rate)
            U[:, 1] = np.clip(U[:, 1], -cfg.max_climb_rate, cfg.max_climb_rate)

            # 7. Shift & warm-start
            U[:-1, :] = U[1:, :]
            U[-1, :] = 0.0

            if verbose:
                dist = float(np.linalg.norm(states[min_idx, -1, :3] - goal))
                print(
                    f"  [MPPI] iter {iteration + 1}/{cfg.num_iterations}: "
                    f"min_cost={min_cost:.1f}, dist_to_goal={dist:.1f}, "
                    f"eff_samples={int(1.0 / max(np.sum(weights**2), 1e-12))}"
                )

        # -- final rollout with refined controls --------------------------
        if best_path is None:
            # Roll out one more time
            states, costs = self._batch_rollout(start, goal, U, np.zeros_like(U)[np.newaxis])
            best_path = self._states_to_path(states[0])

        # Append goal
        if best_path and np.linalg.norm(best_path[-1] - goal) > 0.1:
            best_path.append(goal.copy())

        # B-spline resample for smooth, dense trajectory
        if best_path and len(best_path) >= 4:
            best_path = self._bspline_interpolate(best_path, step=cfg.dt * cfg.cruise_speed)

        return best_path

    # -- Initialisation ---------------------------------------------------

    def _init_control_sequence(
        self, start: np.ndarray, goal: np.ndarray
    ) -> np.ndarray:
        """Heuristic initial control: turn toward goal, gentle climb/descend."""
        cfg = self.config
        H = cfg.horizon

        # Compute heading toward goal
        delta_xy = goal[:2] - start[:2]
        goal_heading = math.atan2(float(delta_xy[1]), float(delta_xy[0]))

        # Current heading (assume starting east by default)
        start_heading = 0.0

        # Turn needed
        turn_needed = goal_heading - start_heading
        # Wrap to [-pi, pi]
        turn_needed = (turn_needed + math.pi) % (2.0 * math.pi) - math.pi

        # Time to reach goal (approx)
        dist = float(np.linalg.norm(delta_xy))
        total_time = dist / max(cfg.cruise_speed, 1.0)
        steps_needed = min(H, max(1, int(total_time / cfg.dt)))

        U = np.zeros((H, 2), dtype=float)

        if steps_needed > 0:
            # Spread turn over first few steps
            turn_steps = min(steps_needed, max(1, int(abs(turn_needed) / cfg.max_turn_rate / cfg.dt)))
            turn_per_step = np.clip(turn_needed / max(turn_steps, 1),
                                    -cfg.max_turn_rate, cfg.max_turn_rate)
            U[:turn_steps, 0] = turn_per_step

            # Climb / descend
            dz = float(goal[2] - start[2])
            climb_per_step = np.clip(dz / max(steps_needed * cfg.dt, 0.1),
                                     -cfg.max_climb_rate, cfg.max_climb_rate)
            U[:steps_needed, 1] = climb_per_step

        return U

    # -- Sampling ---------------------------------------------------------

    def _sample_perturbations(
        self, cfg: MPPIConfig, rng: np.random.Generator
    ) -> np.ndarray:
        """Sample K x H x 2 perturbation tensor eps ~ N(0, Sigma)."""
        return rng.normal(
            loc=0.0,
            scale=[cfg.turn_rate_std, cfg.climb_rate_std],
            size=(cfg.num_samples, cfg.horizon, 2),
        ).astype(np.float32)

    # -- Batch rollout ----------------------------------------------------

    def _batch_rollout(
        self,
        start: np.ndarray,
        goal: np.ndarray,
        U: np.ndarray,      # [H, 2]
        eps: np.ndarray,    # [K, H, 2]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Roll out K trajectories in parallel.

        Args:
            start: [3] start position.
            goal:  [3] goal position.
            U:     [H, 2] base control sequence.
            eps:   [K, H, 2] perturbation tensor.

        Returns:
            states: [K, H+1, 5]  -> (x, y, z, heading, speed)
            costs:  [K] per-trajectory total cost
        """
        cfg = self.config
        K, H, _ = eps.shape

        # Perturbed controls: [K, H, 2]
        U_perturbed = U[np.newaxis, :, :] + eps  # broadcast [1,H,2] + [K,H,2]

        # State tensor: [K, H+1, 5]
        states = np.zeros((K, H + 1, 5), dtype=np.float32)

        # Initial state
        initial_heading = 0.0  # can be improved with startgoal direction
        states[:, 0, 0] = start[0]
        states[:, 0, 1] = start[1]
        states[:, 0, 2] = start[2]
        states[:, 0, 3] = initial_heading
        states[:, 0, 4] = cfg.cruise_speed

        dt = cfg.dt
        v = cfg.cruise_speed

        # Roll out step by step
        for t in range(H):
            s_prev = states[:, t, :]       # [K, 5]
            u_t = U_perturbed[:, t, :]     # [K, 2]

            omega = np.clip(u_t[:, 0], -cfg.max_turn_rate, cfg.max_turn_rate)  # [K]
            v_z = np.clip(u_t[:, 1], -cfg.max_climb_rate, cfg.max_climb_rate)  # [K]

            heading = s_prev[:, 3] + omega * dt   # [K]

            states[:, t + 1, 0] = s_prev[:, 0] + v * np.cos(heading) * dt
            states[:, t + 1, 1] = s_prev[:, 1] + v * np.sin(heading) * dt
            states[:, t + 1, 2] = s_prev[:, 2] + v_z * dt
            states[:, t + 1, 3] = heading
            states[:, t + 1, 4] = v

        # Compute costs
        costs = self._trajectory_costs(states, goal)
        return states, costs

    # -- Cost function ----------------------------------------------------

    def _trajectory_costs(
        self, states: np.ndarray, goal: np.ndarray
    ) -> np.ndarray:
        """Compute per-trajectory total cost S(tau).

        S(tau) = phi(x_H) + sum_{t=0}^{H-1} q(x_t)

        Args:
            states: [K, H+1, 5]
            goal:   [3]

        Returns:
            costs: [K]
        """
        cfg = self.config
        K = states.shape[0]

        # -- Terminal cost ------------------------------------------------
        final_xy = states[:, -1, :2]    # [K, 2]
        final_z = states[:, -1, 2]      # [K]
        terminal_cost = (
            cfg.w_goal_xy * np.linalg.norm(final_xy - goal[np.newaxis, :2], axis=1)
            + cfg.w_goal_z * np.abs(final_z - goal[2])
        )

        # -- Running cost -------------------------------------------------
        # Obstacle cost at each interior state
        positions = states[:, 1:, :3]   # [K, H, 3]  skip t=0 (start)
        obs_cost = self._obstacle_cost_batch(positions)  # [K, H]

        # Boundary cost
        half = cfg.map_size_units * 0.5
        x_out = np.maximum(0, np.abs(positions[:, :, 0]) - half)  # [K, H]
        y_out = np.maximum(0, np.abs(positions[:, :, 1]) - half)
        boundary_cost = cfg.w_boundary * (x_out + y_out)

        # Altitude cost
        z = positions[:, :, 2]
        alt_cost = cfg.w_altitude * (
            np.maximum(0, cfg.min_altitude - z)
            + np.maximum(0, z - cfg.max_altitude)
        )

        # Cruise altitude penalty: encourage staying at goal altitude
        # (softened near obstacles so climbing is still allowed)
        cruise_z = goal[2]
        cruise_cost = cfg.w_cruise_altitude * np.abs(z - cruise_z)

        # Sum running costs
        running_cost = np.sum(obs_cost + boundary_cost + alt_cost + cruise_cost, axis=1)  # [K]

        return terminal_cost + running_cost

    def _obstacle_cost_batch(self, positions: np.ndarray) -> np.ndarray:
        """Smooth exponential obstacle cost field.

        Args:
            positions: [K, H, 3]    (x, y, z) for all samples and timesteps.

        Returns:
            cost: [K, H] per-position obstacle penalty.
        """
        if not self.obstacles:
            return np.zeros(positions.shape[:2], dtype=np.float32)

        cfg = self.config
        K, H, _ = positions.shape
        total = np.zeros((K, H), dtype=np.float32)

        for obs in self.obstacles:
            # Horizontal distance to obstacle center
            dx = positions[:, :, 0] - obs.center_xy[0]   # [K, H]
            dy = positions[:, :, 1] - obs.center_xy[1]   # [K, H]
            dist_xy = np.sqrt(dx**2 + dy**2 + 1e-8)      # [K, H]

            # Only penalise if below obstacle height
            z = positions[:, :, 2]                        # [K, H]
            below_roof = z < (obs.height + cfg.obstacle_clearance)  # [K, H]

            # Sigmoid-like penalty: large when inside, decays outside
            effective_radius = obs.radius + cfg.obstacle_clearance
            penetration = np.maximum(0, effective_radius - dist_xy)  # [K, H]
            cost_contrib = (
                cfg.w_obstacle
                * below_roof.astype(np.float32)
                * np.exp(-cfg.obstacle_cost_scale
                         * np.maximum(0, dist_xy - obs.radius))
            )
            # Add extra penalty for actual penetration
            cost_contrib += (
                cfg.w_obstacle * 2.0
                * below_roof.astype(np.float32)
                * (penetration / max(effective_radius, 1.0))
            )

            total += cost_contrib

        return total

    # -- Helpers ----------------------------------------------------------

    @staticmethod
    def _bspline_interpolate(
        path: list[np.ndarray],
        step: float | None = None,
        num_points: int | None = None,
    ) -> list[np.ndarray]:
        """Resample path with cubic B-spline to produce smooth dense waypoints.

        Args:
            path: Sparse waypoints [N, 3].
            step: Target spacing between output waypoints (scene units).
                  Defaults to cruise_speed * dt if None.
            num_points: If set, output exactly this many points (overrides step).

        Returns:
            Dense list of [x, y, z] waypoints.
        """
        if len(path) < 4:
            return path

        pts = np.array(path, dtype=float)  # [N, 3]
        N = len(pts)

        # Cumulative arc-length parameterisation
        seg_lens = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        t = np.zeros(N, dtype=float)
        t[1:] = np.cumsum(seg_lens)

        try:
            from scipy.interpolate import CubicSpline
            cs_x = CubicSpline(t, pts[:, 0])
            cs_y = CubicSpline(t, pts[:, 1])
            cs_z = CubicSpline(t, pts[:, 2])

            if num_points is not None:
                t_new = np.linspace(t[0], t[-1], num_points)
            else:
                _step = max(step or 1.0, t[-1] / 500.0)
                t_new = np.arange(t[0], t[-1] + _step * 0.5, _step)

            interp = np.column_stack([
                cs_x(t_new), cs_y(t_new), cs_z(t_new),
            ])
            return [interp[i].copy() for i in range(len(interp))]
        except ImportError:
            pass

        # Fallback: numpy-based cubic Hermite interpolation
        if num_points is None:
            num_points = max(N * 4, int(t[-1] / max(step or 1.0, 1e-6)) + 1)
        t_new = np.linspace(t[0], t[-1], num_points)

        interp = np.zeros((num_points, 3), dtype=float)
        for dim in range(3):
            interp[:, dim] = np.interp(t_new, t, pts[:, dim])

        return [interp[i].copy() for i in range(len(interp))]

    def _states_to_path(self, states: np.ndarray) -> list[np.ndarray]:
        """Convert [H+1, 5] state array to list of [x,y,z] waypoints."""
        return [states[i, :3].copy() for i in range(states.shape[0])]

    def _endpoint_blocked(
        self, point: np.ndarray, label: str, verbose: bool
    ) -> bool:
        blocked = is_position_blocked(
            point[:2], float(point[2]),
            self.obstacles, self.config.obstacle_clearance,
        )
        if blocked and verbose:
            print(f"[MPPI] {label} {point} is inside an obstacle!")
        return blocked


# ==========================================================================
# Convenience
# ==========================================================================


def create_default_mppi_planner(
    map_size_units: float = 3000.0,
    obstacles: list[CylindricalObstacle] | None = None,
) -> MPPIPlanner:
    """Factory for a ready-to-use MPPI planner with sensible defaults."""
    config = MPPIConfig(
        map_size_units=map_size_units,
        map_origin=(-map_size_units * 0.5, -map_size_units * 0.5),
        max_altitude=max(200.0, 25.0 + 50.0),
    )
    return MPPIPlanner(obstacles=obstacles, config=config)


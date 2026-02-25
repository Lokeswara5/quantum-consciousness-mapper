import unittest
import numpy as np
from src.core.hyperdimensional_analyzer import HyperDimensionalAnalyzer, HyperDimensionalState

class TestStatePrediction(unittest.TestCase):
    def setUp(self):
        self.analyzer = HyperDimensionalAnalyzer(dimensions=3)
        self.num_points = 5
        self.timesteps = 5

    def create_test_trajectory(self, pattern_type="linear"):
        """Creates test states with known evolution patterns"""
        states = []
        t = np.linspace(0, 1, self.timesteps)
        dt = t[1] - t[0]

        if pattern_type == "linear":
            # Linear motion with velocity-dependent acceleration
            base_velocity = 0.2  # Base velocity
            base_acceleration = 0.05   # Base acceleration

            for step in range(self.timesteps):
                coords = np.zeros((self.num_points, 3))
                velocity = np.zeros((self.num_points, 3))
                accel = np.zeros((self.num_points, 3))

                # Position with uniform spacing and velocity-dependent acceleration
                time = step * dt
                for i in range(self.num_points):
                    x0 = i * 0.25  # Initial spacing
                    v0 = base_velocity * (1 + 0.1 * i)  # Slightly different velocities
                    a0 = base_acceleration * (1 + 0.1 * i)  # Acceleration scales with velocity
                    coords[i, 0] = x0 + v0 * time + 0.5 * a0 * time * time
                    velocity[i, 0] = v0 + a0 * time
                    accel[i, 0] = a0

                states.append(HyperDimensionalState(
                    coordinates=coords.copy(),
                    velocity=velocity.copy(),
                    acceleration=accel.copy(),
                    phase_space_density=1.0,
                    topological_features=[],
                    emergent_patterns=[]
                ))

        elif pattern_type == "circular":
            # Circular motion with constant angular velocity
            omega = 2.0  # Angular velocity
            radius = 1.0  # Fixed radius

            for step in range(self.timesteps):
                coords = np.zeros((self.num_points, 3))
                velocity = np.zeros((self.num_points, 3))
                accel = np.zeros((self.num_points, 3))

                # Create points evenly spaced around circle
                base_angles = np.linspace(0, 2*np.pi, self.num_points, endpoint=False)
                time = step * dt
                angles = base_angles + omega * time

                # Position
                coords[:, 0] = radius * np.cos(angles)
                coords[:, 1] = radius * np.sin(angles)

                # Velocity (tangential)
                velocity[:, 0] = -omega * radius * np.sin(angles)
                velocity[:, 1] = omega * radius * np.cos(angles)

                # Acceleration (centripetal)
                accel[:, 0] = -omega * omega * radius * np.cos(angles)
                accel[:, 1] = -omega * omega * radius * np.sin(angles)

                states.append(HyperDimensionalState(
                    coordinates=coords.copy(),
                    velocity=velocity.copy(),
                    acceleration=accel.copy(),
                    phase_space_density=1.0,
                    topological_features=[],
                    emergent_patterns=[]
                ))

        elif pattern_type == "spiral":
            # Spiral motion with consistent radial expansion
            omega = 2.0  # Angular velocity
            radial_velocity = 0.1  # Outward velocity

            for step in range(self.timesteps):
                coords = np.zeros((self.num_points, 3))
                velocity = np.zeros((self.num_points, 3))
                accel = np.zeros((self.num_points, 3))

                # Base angles evenly spaced
                base_angles = np.linspace(0, 2*np.pi, self.num_points, endpoint=False)
                time = step * dt
                angles = base_angles + omega * time

                # Radius increases with time and varies with angle
                radius = 1.0 + radial_velocity * time
                radius_variation = 0.2 * np.cos(3*base_angles)
                current_radius = radius * (1 + radius_variation)

                # Position
                coords[:, 0] = current_radius * np.cos(angles)
                coords[:, 1] = current_radius * np.sin(angles)
                coords[:, 2] = 0.1 * angles

                # Velocity
                velocity[:, 0] = (radial_velocity * np.cos(angles) -
                                omega * current_radius * np.sin(angles))
                velocity[:, 1] = (radial_velocity * np.sin(angles) +
                                omega * current_radius * np.cos(angles))
                velocity[:, 2] = 0.1 * omega

                # Acceleration (centripetal + radial)
                accel[:, 0] = -omega * omega * current_radius * np.cos(angles)
                accel[:, 1] = -omega * omega * current_radius * np.sin(angles)

                states.append(HyperDimensionalState(
                    coordinates=coords.copy(),
                    velocity=velocity.copy(),
                    acceleration=accel.copy(),
                    phase_space_density=1.0,
                    topological_features=[],
                    emergent_patterns=[]
                ))

        return states

    def test_linear_prediction(self):
        """Test prediction of linear motion"""
        # Create linear trajectory
        states = self.create_test_trajectory("linear")
        self.analyzer.state_history = states

        # Get time parameters from trajectory creation
        t = np.linspace(0, 1, self.timesteps)
        dt = t[1] - t[0]

        # Get current state
        current_state = states[-1]
        initial_positions = current_state.coordinates[:, 0]
        initial_velocities = current_state.velocity[:, 0]
        initial_accelerations = current_state.acceleration[:, 0]

        # Print test parameters
        print("\nTest setup:")
        print(f"dt = {dt}")
        print(f"timesteps = {self.timesteps}")

        # Print current state values
        print("\nCurrent state values (point 0):")
        print(f"Position: {initial_positions[0]}")
        print(f"Velocity: {initial_velocities[0]}")
        print(f"Acceleration: {initial_accelerations[0]}")

        # Predict future states
        future_states = self.analyzer.predict_consciousness_evolution(timesteps=3)
        self.assertEqual(len(future_states), 3)

        # Verify continuity of motion
        next_state = future_states[0]

        # Print prediction values
        print("\nPrediction values (point 0):")
        print(f"Next position: {next_state.coordinates[0, 0]}")
        print(f"Next velocity: {next_state.velocity[0, 0]}")
        expected_vel = initial_velocities[0] + initial_accelerations[0] * dt
        print(f"Expected velocity: {expected_vel}")
        print(f"Difference: {next_state.velocity[0, 0] - expected_vel}")

        for i in range(len(initial_positions)):
            # Each point should continue with its individual motion
            expected_pos = (initial_positions[i] +
                          initial_velocities[i] * dt +
                          0.5 * initial_accelerations[i] * dt * dt)

            self.assertAlmostEqual(
                next_state.coordinates[i, 0],
                expected_pos,
                places=2,
                msg=f"Point {i} motion prediction failed"
            )

            # Velocity should change according to acceleration
            expected_vel = initial_velocities[i] + initial_accelerations[i] * dt
            self.assertAlmostEqual(
                next_state.velocity[i, 0],
                expected_vel,
                places=2,
                msg=f"Point {i} velocity prediction failed"
            )

    def test_circular_prediction(self):
        """Test prediction of circular motion"""
        # Create circular trajectory
        states = self.create_test_trajectory("circular")
        self.analyzer.state_history = states

        # Predict next state
        future_states = self.analyzer.predict_consciousness_evolution(timesteps=3)
        self.assertEqual(len(future_states), 3)

        # Check first predicted state
        predicted_state = future_states[0]
        last_state = states[-1]

        # Check radius preservation
        expected_radius = np.mean(np.linalg.norm(last_state.coordinates[:, :2], axis=1))
        predicted_radii = np.linalg.norm(predicted_state.coordinates[:, :2], axis=1)
        np.testing.assert_allclose(
            predicted_radii,
            expected_radius * np.ones_like(predicted_radii),
            rtol=0.05,  # Allow 5% variation
            err_msg="Radius not preserved in circular motion"
        )

        # Check velocity perpendicularity
        # Normalize vectors for numerical stability
        pos_norms = np.linalg.norm(predicted_state.coordinates[:, :2], axis=1)
        vel_norms = np.linalg.norm(predicted_state.velocity[:, :2], axis=1)
        normalized_pos = predicted_state.coordinates[:, :2] / pos_norms[:, np.newaxis]
        normalized_vel = predicted_state.velocity[:, :2] / vel_norms[:, np.newaxis]

        # Calculate dot products
        dot_products = np.sum(normalized_pos * normalized_vel, axis=1)
        np.testing.assert_array_almost_equal(
            dot_products,
            np.zeros_like(dot_products),
            decimal=1,  # Allow some deviation due to numerical effects
            err_msg="Velocity not perpendicular to position in circular motion"
        )

    def test_pattern_preservation(self):
        """Test that predictions preserve pattern characteristics"""
        # Create spiral trajectory
        states = self.create_test_trajectory("spiral")
        self.analyzer.state_history = states

        # Get initial pattern characteristics
        initial_state = states[-1]
        center = np.mean(initial_state.coordinates, axis=0)

        # Calculate geometric properties
        rel_positions = initial_state.coordinates - center
        radii = np.linalg.norm(rel_positions, axis=1)
        radial_spread = np.std(radii) / np.mean(radii)  # Normalized spread

        # Calculate kinematic properties
        velocities = initial_state.velocity
        speed_variation = np.std(np.linalg.norm(velocities, axis=1))

        # Calculate dynamic properties
        angular_momenta = np.array([
            np.cross(pos, vel)
            for pos, vel in zip(rel_positions, velocities)
        ])
        ang_momentum_magnitude = np.mean(np.linalg.norm(angular_momenta, axis=1))

        # Predict future states
        future_states = self.analyzer.predict_consciousness_evolution(timesteps=3)
        predicted_state = future_states[0]

        # Calculate predicted properties
        pred_center = np.mean(predicted_state.coordinates, axis=0)
        pred_rel_positions = predicted_state.coordinates - pred_center
        pred_radii = np.linalg.norm(pred_rel_positions, axis=1)
        pred_radial_spread = np.std(pred_radii) / np.mean(pred_radii)

        pred_velocities = predicted_state.velocity
        pred_speed_variation = np.std(np.linalg.norm(pred_velocities, axis=1))

        pred_angular_momenta = np.array([
            np.cross(pos, vel)
            for pos, vel in zip(pred_rel_positions, pred_velocities)
        ])
        pred_ang_momentum_magnitude = np.mean(
            np.linalg.norm(pred_angular_momenta, axis=1)
        )

        # Test preservation of pattern characteristics
        # Allow larger tolerance for more complex properties
        self.assertLess(
            abs(pred_radial_spread - radial_spread),
            0.3 * radial_spread,
            "Spatial distribution not preserved"
        )

        # Normalized differences to handle scale variations
        speed_diff = abs(pred_speed_variation - speed_variation)
        speed_scale = max(speed_variation, 0.1)  # Avoid division by zero
        self.assertLess(
            speed_diff / speed_scale,
            0.3,
            "Velocity distribution not preserved"
        )

        # Angular momentum should be approximately conserved
        momentum_diff = abs(
            pred_ang_momentum_magnitude - ang_momentum_magnitude
        )
        momentum_scale = max(ang_momentum_magnitude, 0.1)
        self.assertLess(
            momentum_diff / momentum_scale,
            0.3,
            "Angular momentum not preserved"
        )

    def test_prediction_uncertainty(self):
        """Test that prediction uncertainty increases with time"""
        # Create spiral trajectory with some noise
        states = self.create_test_trajectory("spiral")

        # Add small random noise
        for state in states:
            noise = np.random.normal(0, 0.01, state.coordinates.shape)
            state.coordinates += noise

        self.analyzer.state_history = states

        # Predict multiple steps into future
        future_states = self.analyzer.predict_consciousness_evolution(timesteps=5)

        # Calculate prediction error at each timestep
        errors = []
        for i, state in enumerate(future_states[:-1]):
            next_state = future_states[i + 1]
            error = np.mean(np.linalg.norm(
                next_state.coordinates - state.coordinates,
                axis=1
            ))
            errors.append(error)

        # Verify errors generally increase with time
        self.assertTrue(np.all(np.diff(errors) >= -0.1))  # Allow small decreases

    def test_invalid_predictions(self):
        """Test prediction behavior with invalid inputs"""
        # Test with no history
        self.analyzer.state_history = []
        future_states = self.analyzer.predict_consciousness_evolution(timesteps=3)
        self.assertEqual(len(future_states), 0)

        # Test with negative timesteps
        states = self.create_test_trajectory("linear")
        self.analyzer.state_history = states
        future_states = self.analyzer.predict_consciousness_evolution(timesteps=-1)
        self.assertEqual(len(future_states), 0)

if __name__ == '__main__':
    unittest.main()
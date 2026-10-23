"""Check that checkpoint filtering and result validation cannot hide invalid inputs."""

import unittest

from pointnav_common import mean_metrics, policy_state_dict


class PointNavHelpersTest(unittest.TestCase):
    def test_policy_keeps_actor_critic_and_encoder(self) -> None:
        original = {"actor_critic.net.encoder": object(), "actor_critic.critic": object()}
        result = policy_state_dict(original)
        self.assertIs(result["net.encoder"], original["actor_critic.net.encoder"])
        self.assertIs(result["critic"], original["actor_critic.critic"])
        for invalid in ({}, {"net.encoder": 1}, {"actor_critic.net": 1, "other": 2}):
            with self.assertRaises(ValueError):
                policy_state_dict(invalid)

    def test_metrics_reject_incomplete_or_nan(self) -> None:
        valid = {"success": 1, "spl": 0.5, "distance_to_goal": 0.1, "num_steps": 8}
        self.assertEqual(mean_metrics([valid])["spl"], 0.5)
        with self.assertRaises(KeyError):
            mean_metrics([{"success": 1}])
        with self.assertRaises(ValueError):
            mean_metrics([{**valid, "spl": float("nan")}])


if __name__ == "__main__":
    unittest.main()

"""Unit tests for diversity framework components."""

import unittest
import numpy as np
import tempfile
import shutil
from diversity.metrics import action_correlation, return_correlation, trade_overlap, similarity_score
from diversity.penalty import DiversityScheduler, compute_penalty, adjust_reward
from diversity.storage import ExpertTrajectoryStorage
from diversity.diversity_manager import DiversityManager
from diversity.reward_wrapper import DiversityRewardWrapper


class TestMetrics(unittest.TestCase):
    def test_action_correlation(self):
        a = np.array([1,2,3,4])
        b = np.array([2,4,6,8])
        corr = action_correlation(a, b)
        self.assertAlmostEqual(corr, 1.0, places=5)
        
        b = np.array([-1,-2,-3,-4])
        corr = action_correlation(a, b)
        self.assertAlmostEqual(corr, 1.0, places=5)
        
        corr = action_correlation(a, np.ones(4))
        self.assertEqual(corr, 0.0)
    
    def test_trade_overlap(self):
        a = np.array([0.5, 0.5, 0.0, -0.5])
        b = np.array([0.6, 0.0, 0.0, -0.6])
        overlap = trade_overlap(a, b, threshold=0.01)
        self.assertAlmostEqual(overlap, 0.5, places=5)
        
        a = np.array([1,1,0,-1])
        b = np.array([1,0,0,-1])
        overlap = trade_overlap(a, b)
        self.assertAlmostEqual(overlap, 0.5, places=5)
    
    def test_similarity_score(self):
        score = similarity_score(0.5, 0.6, 0.7)
        expected = 0.4*0.5 + 0.4*0.6 + 0.2*0.7
        self.assertAlmostEqual(score, expected)
        with self.assertRaises(ValueError):
            similarity_score(0.5, 0.5, 0.5, weights=(0.5,0.5,0.5))


class TestScheduler(unittest.TestCase):
    def test_schedule(self):
        schedule = [(1, 0.0), (21, 0.002), (41, 0.005)]
        sch = DiversityScheduler(schedule=schedule)
        sch.update_epoch(10)
        self.assertEqual(sch.lambda_value, 0.0)
        sch.update_epoch(25)
        self.assertEqual(sch.lambda_value, 0.002)
        sch.update_epoch(50)
        self.assertEqual(sch.lambda_value, 0.005)


class TestStorage(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.storage = ExpertTrajectoryStorage(self.temp_dir)
    
    def tearDown(self):
        shutil.rmtree(self.temp_dir)
    
    def test_save_load(self):
        traj = {'actions': np.array([1,2,3]), 'returns': np.array([0.1,0.2,0.3]),
                'positions': np.array([0,1,0])}
        self.storage.save("test_expert", traj)
        loaded = self.storage.load("test_expert")
        self.assertIsNotNone(loaded)
        np.testing.assert_array_equal(loaded['actions'], traj['actions'])
    
    def test_missing(self):
        self.assertIsNone(self.storage.load("missing"))


class TestDiversityManager(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        # Schedule: λ = 0.0 for epochs < 10, then 0.005 for epochs >= 10
        self.manager = DiversityManager(storage_dir=self.temp_dir,
                                        schedule=[(1, 0.0), (10, 0.005)])
        ppo_traj = {'actions': np.array([1,1,0,-1]), 'returns': np.array([0.1,0.1,0.0,-0.1]),
                    'positions': np.array([1,1,0,-1])}
        self.manager.register_expert("PPO", ppo_traj)
    
    def tearDown(self):
        shutil.rmtree(self.temp_dir)
    
    def test_compute_penalty(self):
        # At epoch 5, λ = 0.0 → penalty must be 0.0
        self.manager.update_epoch(5)
        current_actions = np.array([1,0,0,-1])
        current_returns = np.array([0.1,0.0,0.0,-0.1])
        current_positions = np.array([1,0,0,-1])
        penalty = self.manager.compute_penalty(current_actions, current_returns,
                                               current_positions, "SAC")
        self.assertEqual(penalty, 0.0)
        self.assertEqual(self.manager.get_current_lambda(), 0.0)
        
        # At epoch 10, λ = 0.005 → penalty should be > 0.0 (similarity > 0)
        self.manager.update_epoch(10)
        penalty2 = self.manager.compute_penalty(current_actions, current_returns,
                                                current_positions, "SAC")
        self.assertGreater(penalty2, 0.0)
        self.assertGreater(penalty2, penalty)


class TestRewardWrapper(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.manager = DiversityManager(storage_dir=self.temp_dir)
        ppo_traj = {'actions': np.array([1,1,0,-1]), 'returns': np.array([0.1,0.1,0.0,-0.1]),
                    'positions': np.array([1,1,0,-1])}
        self.manager.register_expert("PPO", ppo_traj)
        self.wrapper = DiversityRewardWrapper(self.manager, "SAC")
    
    def tearDown(self):
        shutil.rmtree(self.temp_dir)
    
    def test_adjustment(self):
        self.wrapper.add_step(0.5, 0.01, 0.5)
        self.wrapper.add_step(0.6, 0.02, 0.6)
        penalty = self.wrapper.compute_penalty_for_batch()
        trading_reward = 0.05
        final = self.wrapper.adjust_reward(trading_reward, penalty)
        self.assertEqual(final, trading_reward - penalty)
        self.wrapper.reset()
        self.assertEqual(len(self.wrapper._action_buffer), 0)


if __name__ == '__main__':
    unittest.main()
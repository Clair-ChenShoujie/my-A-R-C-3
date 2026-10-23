import asyncio
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from arc3.config import Settings
from arc3.experiment import apply_development_budget, select_game_ids
from arc3.benchmark import execute
from arc3.__main__ import run

ROOT = Path(__file__).resolve().parents[1]

class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.settings = Settings.load(ROOT / 'configs/baseline.json')

    def test_ambiguous_game_prefix_is_rejected(self):
        with self.assertRaises(ValueError):
            select_game_ids(['lf52-a', 'lf52-b'], ['lf52'])
        self.assertEqual(select_game_ids(['lf52-a', 'tn36-b'], ['lf52']), ['lf52-a'])

    def test_budget_accounts_for_selected_games_and_passes(self):
        bm = SimpleNamespace(solver=SimpleNamespace())
        with patch.dict(os.environ):
            apply_development_budget(bm, dict(profile='noop_recovery_v91', minutes=5,
                                     active_streams=10, passes=2), game_count=3)
            self.assertEqual(bm.solver.concurrency, 6)
            self.assertEqual(os.environ['ARC3_MAX_ACTIVE_STREAMS'], '6')
            self.assertEqual(bm.solver.max_runtime_s_per_game, 300)
            self.assertEqual(bm.n_passes, 2)

    def test_invalid_graph_capacity_fails_before_model_work(self):
        data = json.loads((ROOT / 'configs/baseline.json').read_text())
        data['inference']['CUDAGRAPH_MAXBS'] = 4
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / 'bad.json'
            p.write_text(json.dumps(data))
            with self.assertRaises(ValueError):
                Settings.load(p)

    def test_smoke_routes_to_campaign_with_project_worker(self):
        from arc3 import experiment
        with tempfile.TemporaryDirectory() as tmp:
            ctx = SimpleNamespace(settings=self.settings, TRUE_SUBMISSION=False,
                BUNDLE_DIR=Path(tmp), WORKING_DIR=Path(tmp), NOTEBOOK_START_EPOCH=0,
                SERVER_BASE_URL='http://127.0.0.1:8001/v1', ACTIVE_PROFILE='noop_recovery_v91',
                EXPERIMENT_PROFILES=experiment.PROFILES, USE_PRIORITY_SCHEDULING=True)
            pandas = SimpleNamespace(DataFrame=Mock(return_value=SimpleNamespace(to_parquet=Mock())))
            with patch.object(experiment, 'run_campaign', new_callable=AsyncMock) as campaign, \
                 patch.dict(sys.modules, {'pandas': pandas}):
                asyncio.run(execute(ctx))
                kw = campaign.call_args.kwargs
                self.assertEqual(kw['minutes'], 5)
                self.assertEqual(kw['active_streams'], 1)
                self.assertEqual(kw['games'], ['lf52'])
                self.assertEqual(kw['helper_path'], Path(experiment.__file__))

    def test_runner_stops_monitor_and_server_after_benchmark_failure(self):
        ctx = SimpleNamespace(TRUE_SUBMISSION=False)
        settings = Mock()
        proc = Mock(pid=123456)
        proc.poll.return_value = None
        stop = Mock()
        with patch('arc3.environment.prepare', return_value=ctx), \
             patch('arc3.precache.prepare_precache'), patch('arc3.runtime.install'), \
             patch('arc3.serving.launch', return_value=proc), \
             patch('arc3.monitor.start_monitor', return_value=stop), \
             patch('arc3.monitor._sysline', return_value='test monitor'), \
             patch('arc3.benchmark.execute', new_callable=AsyncMock, side_effect=RuntimeError('boom')), \
             patch('os.killpg') as kill:
            with self.assertRaisesRegex(RuntimeError, 'boom'):
                asyncio.run(run(settings))
            stop.set.assert_called_once()
            kill.assert_called_once()
            proc.wait.assert_called_once_with(timeout=20)

    def test_cli_check_works_without_kaggle_inputs(self):
        result = subprocess.run([sys.executable, '-m', 'arc3', '--config',
                                 str(ROOT / 'configs/baseline.json'), '--check'],
                                capture_output=True, text=True, check=True)
        self.assertEqual(json.loads(result.stdout)['status'], 'config-valid')

if __name__ == '__main__':
    unittest.main()

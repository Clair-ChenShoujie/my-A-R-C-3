"""Run development campaigns or the competition benchmark."""
import os
import pickle
import time
from pathlib import Path
from datetime import datetime, timedelta
from urllib.request import urlopen
from . import experiment

async def execute(ctx):
    TRUE_SUBMISSION = ctx.TRUE_SUBMISSION
    BUNDLE_DIR = ctx.BUNDLE_DIR
    WORKING_DIR = ctx.WORKING_DIR
    NOTEBOOK_START_EPOCH = ctx.NOTEBOOK_START_EPOCH
    SERVER_BASE_URL = ctx.SERVER_BASE_URL
    ACTIVE_PROFILE = ctx.ACTIVE_PROFILE
    EXPERIMENT_PROFILES = ctx.EXPERIMENT_PROFILES
    USE_PRIORITY_SCHEDULING = ctx.USE_PRIORITY_SCHEDULING
    DEV_MODE = ctx.settings.run['DEV_MODE']
    SUBMISSION_PROFILE = ctx.settings.run['SUBMISSION_PROFILE']
    DEV_PROFILES = ctx.settings.run['DEV_PROFILES']
    DEV_GAME_PREFIXES = ctx.settings.run['DEV_GAME_PREFIXES']
    DEV_MINUTES_PER_ARM = ctx.settings.run['DEV_MINUTES_PER_ARM']
    DEV_REPEATS = ctx.settings.run['DEV_REPEATS']
    DEV_ACTIVE_STREAMS = ctx.settings.run['DEV_ACTIVE_STREAMS']
    DEV_SAVE_REQUEST_LOGS = ctx.settings.run['DEV_SAVE_REQUEST_LOGS']
    DEV_FULL_PROFILE = ctx.settings.run['DEV_FULL_PROFILE']
    helper_path = Path(experiment.__file__)

    if TRUE_SUBMISSION or DEV_MODE == "full":
        # Restore the deployment target and record the real submission state on it.
        with open(BUNDLE_DIR / "deploy_target.pkl", "rb") as file:
            target = pickle.load(file)
        target.actual_run_as_submission = TRUE_SUBMISSION
        target.is_competition_rerun = TRUE_SUBMISSION

        # Restore the benchmark and point its outputs at the Kaggle working dir.
        with open(BUNDLE_DIR / "benchmark_initial.pkl", "rb") as file:
            bm = pickle.load(file)
        bm.job_dir = WORKING_DIR
    else:
        print("Short tests restore fresh benchmarks inside separate worker processes.")


    
    if TRUE_SUBMISSION or DEV_MODE == "full":
        # Make one-off changes to `bm`, `bm.games`, or `bm.solver` here before the run starts.
        bm.solver.max_runtime_s_per_game = 48*60 #7920.0
        bm.solver.analyzer_timeout = 900.0
        bm.solver.concurrency = 10
        bm.solver.max_actions_per_game = None
        bm.solver.save_request_logs = True
        bm.n_passes = 4
        bm.game_weights = None

        demo_excluded_games = [] if TRUE_SUBMISSION else []

        if TRUE_SUBMISSION:  # overrides for contest run
            bm.solver.save_request_logs = False
            bm.n_passes = 1
            if USE_PRIORITY_SCHEDULING:
                bm.solver.concurrency = 120  # game count should be 110
                bm.solver.max_runtime_s_per_game = 532*60
        else:
            if USE_PRIORITY_SCHEDULING:
                bm.solver.concurrency = (25 - len(demo_excluded_games)) * bm.n_passes
                bm.solver.max_runtime_s_per_game = 532*60 * bm.solver.concurrency // 110
            if float(getattr(target, 'max_runtime_s', 0.0) or 0.0) != 32400.0:
                raise RuntimeError(
                    f'Expected the 32400-second notebook budget, got {target.max_runtime_s!r}.'
                )

        # show config
        print('benchmark.solver:', bm.solver)
    else:
        print("Short-test budgets are set AFTER game selection in each worker; no long-budget override applies.")


        assert EXPERIMENT_PROFILES == experiment.PROFILES

    if not TRUE_SUBMISSION and DEV_MODE != "full":
        EXPERIMENT_DIR = await experiment.run_campaign(
            helper_path=helper_path, bundle_dir=BUNDLE_DIR, working_dir=WORKING_DIR,
            environment_dir="/kaggle/input/competitions/arc-prize-2026-arc-agi-3/environment_files",
            base_url=SERVER_BASE_URL, mode=DEV_MODE, profiles=DEV_PROFILES,
            games=DEV_GAME_PREFIXES, minutes=DEV_MINUTES_PER_ARM,
            repeats=DEV_REPEATS, active_streams=DEV_ACTIVE_STREAMS,
            save_request_logs=DEV_SAVE_REQUEST_LOGS,
        )
        # Same offline placeholder as the original notebook. Real reruns use the gateway output.
        import pandas as pd
        pd.DataFrame([["1_0", "1", True, 1]],
                     columns=["row_id", "game_id", "end_of_game", "score"]).to_parquet(
                         WORKING_DIR / "submission.parquet", index=False)
    else:
        print('Starting benchmark...')

        # Build the live competition game list from the gateway's available environments.
        def _competition_games():
            import arc_agi

            import taaf.game_api

            spec = taaf.game_api.ArcadeSpec(
                operation_mode=arc_agi.OperationMode.COMPETITION,
                arc_base_url=os.environ["ARC_BASE_URL"],
                environments_dir="",
            )
            arcade = arc_agi.Arcade(
                operation_mode=arc_agi.OperationMode.COMPETITION,
                arc_base_url=spec.arc_base_url,
                environments_dir="",
            )
            game_ids = [env_info.game_id for env_info in arcade.available_environments]
            if not game_ids:
                raise RuntimeError("Competition Arcade exposed zero environments.")
            return [taaf.game_api.GameAPI(env_name=game_id, arcade_spec=spec) for game_id in game_ids]


        # Build the offline game list from the competition's bundled environment files.
        def _offline_games(env_dir: str):
            import arc_agi

            import taaf.game_api

            spec = taaf.game_api.ArcadeSpec(operation_mode=arc_agi.OperationMode.OFFLINE, environments_dir=env_dir)
            arcade = arc_agi.Arcade(operation_mode=arc_agi.OperationMode.OFFLINE, environments_dir=env_dir)
            game_ids = [env_info.game_id for env_info in arcade.available_environments]
            if not game_ids:
                raise RuntimeError(f"No offline environments found under {env_dir}.")
            game_ids = [game_id for game_id in game_ids if not any(game_id.startswith(g) for g in demo_excluded_games)]
            return [taaf.game_api.GameAPI(env_name=game_id, arcade_spec=spec) for game_id in game_ids]


        # The gateway can take a while to come up; poll until it answers.
        def _wait_for_gateway(base_url: str, timeout_s: float = 600.0) -> None:
            deadline = time.monotonic() + timeout_s
            last_error = ""
            while time.monotonic() < deadline:
                try:
                    with urlopen(f"{base_url}api/games", timeout=10) as response:
                        if response.status < 500:
                            return
                except Exception as exc:
                    last_error = repr(exc)
                time.sleep(5)
            raise RuntimeError(f"Kaggle gateway did not become ready: {last_error}")


        # Print the run preamble and persist the launcher's git status for diagnostics.
        #print((BUNDLE_DIR / "preamble.txt").read_text())
        (WORKING_DIR / "git_status.txt").write_text((BUNDLE_DIR / "git_status.txt").read_text())

        # arc_agi reads RECORDINGS_DIR and ARC_API_KEY from env (ArcadeSpec carries neither); operation
        # mode, environments dir, and base url are all passed explicitly via the spec, so no env is needed.
        os.environ.setdefault("RECORDINGS_DIR", str(WORKING_DIR / "server_recording"))

        if TRUE_SUBMISSION:
            # Real submission: play the live competition Arcade served by the Kaggle gateway.
            os.environ.setdefault("ARC_API_KEY", "test-key-123")
            os.environ.setdefault("ARC_BASE_URL", "http://gateway:8001/")
            # The gateway boots asynchronously; wait before swapping in its game list.
            _wait_for_gateway(os.environ["ARC_BASE_URL"])
            bm.games = _competition_games()
        else:
            # Interactive run: play the bundled competition environments offline (no gateway).
            # The competition's environment files ship alongside the wheelhouse in the competition dataset.
            competition_env_files = str(Path("/kaggle/input/competitions/arc-prize-2026-arc-agi-3/arc_agi_3_wheels").parent / "environment_files")
            bm.games = _offline_games(competition_env_files)

        # Outside a real submission, stop ~10 min before the wall-clock budget for a graceful exit.
        soft_end = None
        if not TRUE_SUBMISSION:
            budget = float(getattr(target, "max_runtime_s", 0.0) or 0.0)
            if budget > 0:
                soft_end = datetime.fromtimestamp(NOTEBOOK_START_EPOCH) + timedelta(seconds=budget - min(600.0, budget / 2))

        # Play the benchmark; teardown commands run even if the run raises.
        # Passive telemetry only: no extra model calls/actions, no gameplay changes.
        metrics_dir = WORKING_DIR / "run_metrics" / (time.strftime("%Y%m%dT%H%M%S", time.gmtime()) + "_" + ACTIVE_PROFILE)
        observer = experiment.Telemetry(metrics_dir, ACTIVE_PROFILE).install()
        experiment.write_json(metrics_dir / "effective_config.json", {
            "competition_rerun": TRUE_SUBMISSION, "profile": ACTIVE_PROFILE,
            "profile_flags": EXPERIMENT_PROFILES[ACTIVE_PROFILE],
            "game_count": len(bm.games), "passes": bm.n_passes,
            "solver": repr(bm.solver), "soft_end": str(soft_end),
            "active_streams": os.environ.get("ARC3_MAX_ACTIVE_STREAMS"),
        })
        try:
            await bm.run(soft_end_time=soft_end, runtime_environment=target, minimal_diagnostics=TRUE_SUBMISSION)
        finally:
            observer.restore()
            observer._safe(observer.export)
        if not TRUE_SUBMISSION:
            # An offline run isn't scored, but Kaggle still expects a submission.parquet output.
            import pandas as pd

            pd.DataFrame(
                [["1_0", "1", True, 1]],
                columns=["row_id", "game_id", "end_of_game", "score"],
            ).to_parquet(WORKING_DIR / "submission.parquet", index=False)

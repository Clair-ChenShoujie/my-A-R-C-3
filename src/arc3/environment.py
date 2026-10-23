"""Offline bundle setup, harness patching and profile environment."""
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from . import experiment

ASSETS = Path(__file__).parent / "assets"

def prepare(settings):
    DEV_MODE = settings.run['DEV_MODE']
    SUBMISSION_PROFILE = settings.run['SUBMISSION_PROFILE']
    DEV_PROFILES = settings.run['DEV_PROFILES']
    DEV_GAME_PREFIXES = settings.run['DEV_GAME_PREFIXES']
    DEV_MINUTES_PER_ARM = settings.run['DEV_MINUTES_PER_ARM']
    DEV_REPEATS = settings.run['DEV_REPEATS']
    DEV_ACTIVE_STREAMS = settings.run['DEV_ACTIVE_STREAMS']
    DEV_SAVE_REQUEST_LOGS = settings.run['DEV_SAVE_REQUEST_LOGS']
    DEV_FULL_PROFILE = settings.run['DEV_FULL_PROFILE']
    # Kaggle might cancel the run in no action is done in some timeframe.
    # To prevent this, harness will start up even if backend is not yet ready,
    # and the first started game will perform a single reset action.
    NOTEBOOK_START_TIME = time.time()
    SERVER_STARTUP_TIMEOUT = 12 * 60
    os.environ['ARC3_WARMUP_ACTION_GAMES'] = '1'
    os.environ['ARC3_HTTP_RETRY_INITIAL_SECONDS'] = '900'

    # model and dataset paths
    WHEELHOUSE_DIR = str(settings.paths['wheelhouse'])
    MODEL_DIR = str(settings.paths['model'])
    DRAFT_MODEL_DIR = str(settings.paths['draft_model'])
    ORIG_BUNDLE_DIR = str(settings.paths['source_bundle'])
    SERVED_MODEL_NAME = 'flashnext'

    # Apply harness patch
    BUNDLE_DIR = Path(settings.paths["bundle_work"])
    source = Path(ORIG_BUNDLE_DIR).resolve()
    destination = BUNDLE_DIR.resolve()
    if destination == source or destination in source.parents or source in destination.parents:
        raise ValueError("source_bundle and bundle_work must be separate directories")
    if destination.parent != Path('/kaggle') or destination.name != 'taaf-kaggle-source-share':
        raise ValueError("bundle_work must be the dedicated /kaggle/taaf-kaggle-source-share directory")
    shutil.rmtree(BUNDLE_DIR, ignore_errors=True)
    shutil.copytree(ORIG_BUNDLE_DIR, BUNDLE_DIR)
    subprocess.run(["git", "apply", "--include=ARC3-Inference/*", "-v", str(ASSETS / "harness.patch")], cwd=f"{BUNDLE_DIR}/src", check=True)
    print('harness patch applied successfully')

    # v4: append a small runtime wrapper instead of editing ToolAgent class internals.
    _tool_agent_path = BUNDLE_DIR / "src" / "ARC3-Inference" / "inference" / "agent" / "tool_agent.py"
    _v6_runtime_patch = (ASSETS / 'cycle_recovery.py.txt').read_text()
    with _tool_agent_path.open("a", encoding="utf-8") as _f:
        _f.write(_v6_runtime_patch)
    subprocess.run([sys.executable, "-m", "py_compile", str(_tool_agent_path)], check=True)
    print("v6 cycle-gated wrapper appended; tool_agent.py syntax OK")

    # v9.1 real-action no-op recovery ToolAgent patch.
    _v91_noop_patch = (ASSETS / 'noop_recovery.py.txt').read_text()
    with _tool_agent_path.open("a", encoding="utf-8") as _f:
        _f.write(_v91_noop_patch)
    subprocess.run([sys.executable, "-m", "py_compile", str(_tool_agent_path)], check=True)
    print("v9.1 real-action no-op recovery appended; tool_agent.py syntax OK")

    # True only inside a real competition rerun; switches diagnostics + soft deadline.
    TRUE_SUBMISSION = os.environ.get("KAGGLE_IS_COMPETITION_RERUN", "").strip().lower() in {"1", "true"}
    NOTEBOOK_START_EPOCH = time.time()
    if DEV_MODE not in {"smoke", "compare", "confirm", "full"}:
        raise ValueError("DEV_MODE must be smoke, compare, confirm, or full")
    EXPERIMENT_PROFILES = experiment.PROFILES
    ACTIVE_PROFILE = SUBMISSION_PROFILE if TRUE_SUBMISSION else DEV_FULL_PROFILE
    if ACTIVE_PROFILE not in EXPERIMENT_PROFILES:
        raise ValueError("Unknown selected profile")
    print(f"Experiment mode: {'COMPETITION' if TRUE_SUBMISSION else DEV_MODE}; profile={ACTIVE_PROFILE}")

    # Non-interactive matplotlib backend: diagnostics render plots with no display attached.
    os.environ["MPLBACKEND"] = "Agg"
    # Marks the run as a (real or emulated) submission so the framework + solver can adjust.
    os.environ["TAAF_RUN_AS_SUBMISSION"] = "1" if TRUE_SUBMISSION else "0"
    # In submission, disable the periodic JSON/HTML diagnostics writes and per-frame logging.
    os.environ["TAAF_MINIMAL_DIAGNOSTICS"] = "1" if TRUE_SUBMISSION else "0"
    # Pin arc_agi's cached level_reset_only before its client is built (RESET keeps the level).
    os.environ["ONLY_RESET_LEVELS"] = "true"

    # Everything the run produces is written here.
    WORKING_DIR = Path(settings.paths["working"])
    WORKING_DIR.mkdir(parents=True, exist_ok=True)
    print(f"taaf.kaggle: TRUE_SUBMISSION={TRUE_SUBMISSION}")
    SERVED_MODEL_HOST = '127.0.0.1'
    SERVED_MODEL_PORT = 8001
    SERVER_BASE_URL = f'http://{SERVED_MODEL_HOST}:{SERVED_MODEL_PORT}/v1'
    USE_PRIORITY_SCHEDULING = True

    setup_env = {
        ########## transformers config ##########
        'USE_TF': '0',
        'TRANSFORMERS_NO_TF': '1',
        'TRANSFORMERS_NO_TORCHVISION': '1',

        ########## serving config ##########
        'LOCAL_ANALYZER_BASE_URL': SERVER_BASE_URL,
        'OPENAI_BASE_URL': SERVER_BASE_URL,
        'LOCAL_ANALYZER_PROVIDER': 'vllm',
        'OPENAI_PROVIDER': 'vllm',
        'LOCAL_ANALYZER_MODEL_ID': SERVED_MODEL_NAME,
        'INFERENCE_ANALYZER_MODEL': SERVED_MODEL_NAME,
        'LOCAL_ANALYZER_APP_NAME': 'ARC3 Agent Harness',
    
        ########## analyzer config ##########
        'ARC3_REASONING_HISTORY_KEY': 'reasoning_content',
        'LOCAL_ANALYZER_ENABLE_THINKING': 'true',
        'LOCAL_ANALYZER_TOOL_STEPS': '0',
        'LOCAL_ANALYZER_TOOL_TIMEOUT': '30',
        'LOCAL_ANALYZER_YIELD_SECONDS': '0',
        'LOCAL_ANALYZER_YIELD_TOKENS': '2048',
        'LOCAL_ANALYZER_TEMPERATURE': '0.7',
        'LOCAL_ANALYZER_TOP_P': '0.95',
        'LOCAL_ANALYZER_TOP_K': '20',

        ########## context pruning ##########
        'LOCAL_ANALYZER_CONTEXT_WINDOW': (116+12)*1024,
        'LOCAL_ANALYZER_MAX_OUTPUT': 12*1024,
        'ARC3_CONTEXT_DRAIN_TOKENS': 58*1024,
        'ARC3_HISTORY_ASSISTANT_TURNS': 150,
        'ARC3_HISTORY_TURN_DRAIN': 30,
        'ARC3_HISTORY_DRAIN_COALESCE': '1',

        ########## tool output ##########
        'LOCAL_ANALYZER_TOOL_OUTPUT_TOKENS': '3072',
        'ARC3_MIDDLE_TRUNCATION': '1',
        'ARC3_ACTION_ECHO': '1',
    
        ########## prompt changes ##########
        'ARC3_EXPLAIN_GAMEPLAY_CHANGED': '1',
        'ARC3_NEW_CHANGED_PROMPTS': '1',
        'ARC3_LEVEL_TRANSFER_GUIDANCE': '1',
        'ARC3_ACTION_INFO': '1',
        'ARC3_DEDUPE_MULTICALL_LINE': '1',
        'ARC3_FRAME_DIFF_HINT': '1',

        ########## vision & animations ##########
        'MULTIMODAL_CONTEXT': 'current_grid',
        'MULTIMODAL_UPSCALE': '10',
        'ARC3_DIFF_IMAGE': '1',
        'ARC3_GAMEOVER_DIFF_IMAGE': '1',
        'ARC3_ANIMATION': '1',

        ########## guards ##########
        'ARC3_GUARDS_FROM_LEVEL': 2,
        'ARC3_BATCH_NOOP_BLOCK': '1',
        'ARC3_STALE_STATE_BLOCK': '1',

        ########## world model ##########
        'ARC3_MEMORY_SECTIONS': 'off',

        ########## persistent functions ##########
        'ARC3_PERSISTENT_FUNCTIONS': '1',
        'ARC3_PERSISTENT_FUNCTIONS_SCOPE': 'game',
        'ARC3_PERSISTENT_FUNCTIONS_IMPORTS': '1',

        ########## controls ##########
        'EXPOSE_UNDO': 'on',
    }

    if USE_PRIORITY_SCHEDULING: setup_env.update({
        'ARC3_MAX_ACTIVE_STREAMS': 10,
        'ARC3_PRIORITY_REFRESH_QUEUE': '1',
        'ARC3_PRIORITY_PACE': '0',
        'ARC3_PRIORITY_TAIL_FADE': '1',
        'ARC3_PRIORITY_TAIL_FADE_FRACTION': '0.2',
        'ARC3_PRIORITY_TAIL_LOOKUP': 'remaining',
        'ARC3_PRIORITY_SCORE_NORMALIZATION': '1',
    })

    # Set flags before unpickling/importing the inference framework.
    setup_env.update(EXPERIMENT_PROFILES[ACTIVE_PROFILE])
    os.environ.update({k: str(v) for k,v in setup_env.items()})
    print("Selected profile flags:", EXPERIMENT_PROFILES[ACTIVE_PROFILE])
    print("Effective retained-function scope:", os.environ.get("ARC3_PERSISTENT_FUNCTIONS_SCOPE", "levelup"))
    print("Hypothesis switching:", os.environ.get("ARC3_STAGNATION_HYPOTHESIS_SWITCH", "0"), "thresholds=", os.environ.get("ARC3_STAGNATION_FIRST_ACTIONS", "35"), os.environ.get("ARC3_STAGNATION_SECOND_ACTIONS", "65"))


    
    return SimpleNamespace(settings=settings, TRUE_SUBMISSION=TRUE_SUBMISSION, NOTEBOOK_START_TIME=NOTEBOOK_START_TIME, SERVER_STARTUP_TIMEOUT=SERVER_STARTUP_TIMEOUT, NOTEBOOK_START_EPOCH=NOTEBOOK_START_EPOCH, WHEELHOUSE_DIR=WHEELHOUSE_DIR, MODEL_DIR=MODEL_DIR, DRAFT_MODEL_DIR=DRAFT_MODEL_DIR, SERVED_MODEL_NAME=SERVED_MODEL_NAME, SERVED_MODEL_PORT=SERVED_MODEL_PORT, SERVER_BASE_URL=SERVER_BASE_URL, BUNDLE_DIR=BUNDLE_DIR, WORKING_DIR=WORKING_DIR, USE_PRIORITY_SCHEDULING=USE_PRIORITY_SCHEDULING, ACTIVE_PROFILE=ACTIVE_PROFILE, EXPERIMENT_PROFILES=EXPERIMENT_PROFILES)

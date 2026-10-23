"""Small experimental runner for the uploaded Franzen Milestone 2 notebook.

No extra model calls, environment actions or automatic submission are added.
Only standard-library imports are needed until the existing harness is loaded.
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import functools
import json
import os
import pickle
import re
import statistics
import subprocess
import sys
import threading
import time
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


PROFILES = {'baseline': {'ARC3_STEP_VERIFICATION_HINT': '0', 'ARC3_PERSISTENT_FUNCTIONS_REPAIR_HINTS': '0', 'ARC3_STAGNATION_HYPOTHESIS_SWITCH': '0', 'ARC3_ADAPTIVE_HYPOTHESIS_SWITCH': '0', 'ARC3_CYCLE_HYPOTHESIS_SWITCH': '0', 'ARC3_NOOP_RECOVERY_V91': '0'}, 'repair_only': {'ARC3_STEP_VERIFICATION_HINT': '0', 'ARC3_PERSISTENT_FUNCTIONS_REPAIR_HINTS': '1', 'ARC3_PERSISTENT_FUNCTIONS_SCOPE': 'game', 'ARC3_STAGNATION_HYPOTHESIS_SWITCH': '0', 'ARC3_ADAPTIVE_HYPOTHESIS_SWITCH': '0', 'ARC3_CYCLE_HYPOTHESIS_SWITCH': '0', 'ARC3_NOOP_RECOVERY_V91': '0'}, 'cycle_switch': {'ARC3_STEP_VERIFICATION_HINT': '0', 'ARC3_PERSISTENT_FUNCTIONS_REPAIR_HINTS': '1', 'ARC3_PERSISTENT_FUNCTIONS_SCOPE': 'game', 'ARC3_STAGNATION_HYPOTHESIS_SWITCH': '0', 'ARC3_ADAPTIVE_HYPOTHESIS_SWITCH': '0', 'ARC3_CYCLE_HYPOTHESIS_SWITCH': '1', 'ARC3_CYCLE_MIN_ACTIONS': '30', 'ARC3_CYCLE_LEVEL1_MIN_ACTIONS': '42', 'ARC3_CYCLE_WINDOW_TURNS': '10', 'ARC3_CYCLE_SECOND_GAP': '24', 'ARC3_CYCLE_MIN_REPEAT_RATIO': '0.34', 'ARC3_CYCLE_STRONG_REPEAT_RATIO': '0.50', 'ARC3_NOOP_RECOVERY_V91': '0'}, 'noop_recovery_v91': {'ARC3_STEP_VERIFICATION_HINT': '0', 'ARC3_PERSISTENT_FUNCTIONS_REPAIR_HINTS': '1', 'ARC3_PERSISTENT_FUNCTIONS_SCOPE': 'game', 'ARC3_STAGNATION_HYPOTHESIS_SWITCH': '0', 'ARC3_ADAPTIVE_HYPOTHESIS_SWITCH': '0', 'ARC3_CYCLE_HYPOTHESIS_SWITCH': '0', 'ARC3_SHADOW_PROFILER': '0', 'ARC3_STRATEGY_SHADOW': '0', 'ARC3_NOOP_RECOVERY_V91': '1', 'ARC3_NOOP_WINDOW_ACTIONS': '24', 'ARC3_NOOP_MIN_LEVEL_ACTIONS': '24', 'ARC3_NOOP_MIN_WINDOW_ACTIONS': '16', 'ARC3_NOOP_MIN_RECENT_NOOPS': '4', 'ARC3_NOOP_MIN_RATIO': '0.18'}}


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    temporary.replace(path)


def competition_rerun():
    return os.environ.get("KAGGLE_IS_COMPETITION_RERUN", "").strip().lower() in {"1", "true"}


def select_game_ids(available_ids, prefixes):
    """Exact-or-unique prefix selection; never silently omit a requested game."""
    ids = list(available_ids)
    if not prefixes:
        return ids
    selected = []
    for prefix in prefixes:
        matches = [game_id for game_id in ids if game_id == prefix or game_id.startswith(prefix + "-")]
        if len(matches) != 1:
            raise ValueError(f"Game prefix {prefix!r} matched {len(matches)} games; available: {ids}")
        if matches[0] in selected:
            raise ValueError(f"Duplicate requested game: {matches[0]}")
        selected.append(matches[0])
    return selected


def validate_development_config(config):
    if config["profile"] not in PROFILES:
        raise ValueError("Unknown profile")
    if not 1 <= float(config["minutes"]) <= 180:
        raise ValueError("Development budget must be between 1 and 180 minutes")
    if not 1 <= int(config["active_streams"]) <= 10:
        raise ValueError("Development active_streams must be between 1 and the existing server limit (10)")
    if not 1 <= int(config["passes"]) <= 4:
        raise ValueError("Development passes must be between 1 and 4")


def apply_development_budget(benchmark, config, game_count):
    """Called AFTER game selection. Includes queue time; deadline is run-wide."""
    validate_development_config(config)
    if game_count < 1:
        raise ValueError("No development games selected")
    benchmark.n_passes = int(config["passes"])
    jobs = game_count * benchmark.n_passes
    benchmark.solver.concurrency = jobs
    benchmark.solver.max_runtime_s_per_game = float(config["minutes"]) * 60
    benchmark.solver.max_actions_per_game = None
    benchmark.solver.max_generated_tokens_per_game = None
    benchmark.solver.analyzer_timeout = 900.0
    benchmark.solver.save_request_logs = bool(config.get("save_request_logs", False))
    benchmark.game_weights = None
    # A fresh worker imports the gate for the first time; no singleton is reused.
    os.environ["ARC3_MAX_ACTIVE_STREAMS"] = str(min(jobs, int(config["active_streams"])))
    return datetime.now() + timedelta(minutes=float(config["minutes"]))


class Telemetry:
    """Read-only observers. Return values/exceptions are passed through unchanged.

    One compact JSON record per completed solver session. No grids or prompts.
    RHAE scores come from the framework's [finished] log, never from level reward.
    """

    def __init__(self, directory, profile):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.profile = profile
        self.local = threading.local()
        self.lock = threading.Lock()
        self.records = []
        self.patches = []
        self.observer_failures = 0

    def _safe(self, callback, *args):
        try:
            callback(*args)
        except Exception:
            with self.lock:
                self.observer_failures += 1

    def _wrap(self, cls, name, wrapper):
        original = getattr(cls, name)
        setattr(cls, name, functools.wraps(original)(wrapper(original)))
        self.patches.append((cls, name, original))

    def install(self):
        from inference.framework import solver as solver_module
        from inference.agent import tool_agent as agent_module

        recorder = self

        def play_wrapper(original):
            def wrapped(session, *args, **kwargs):
                previous = getattr(recorder.local, "row", None)
                row = {"start": time.monotonic(), "counts": Counter(), "errors": Counter(),
                       "queue_seconds": 0.0, "first_action_seconds": None, "exception": None}
                recorder.local.row = row
                try:
                    return original(session, *args, **kwargs)
                except BaseException as exc:
                    row["exception"] = type(exc).__name__
                    raise
                finally:
                    recorder._safe(recorder.finish, session, row)
                    recorder.local.row = previous
            return wrapped

        def action_wrapper(original):
            def wrapped(session, *args, **kwargs):
                result = original(session, *args, **kwargs)
                recorder._safe(recorder.publish_noop_action, session, result)
                recorder._safe(recorder.observe_action, result)
                return result
            return wrapped

        def dispatch_wrapper(original):
            def wrapped(agent, *args, **kwargs):
                result = original(agent, *args, **kwargs)
                recorder._safe(recorder.observe_tool, result)
                return result
            return wrapped

        def queue_wrapper(original):
            def wrapped(gate, *args, **kwargs):
                start = time.monotonic()
                try:
                    return original(gate, *args, **kwargs)
                finally:
                    row = getattr(recorder.local, "row", None)
                    if row is not None:
                        row["queue_seconds"] += time.monotonic() - start
            return wrapped

        self._wrap(solver_module._HarnessGameSession, "play", play_wrapper)
        self._wrap(solver_module._HarnessGameSession, "_execute_action", action_wrapper)
        self._wrap(agent_module.ToolAgent, "_dispatch_tool", dispatch_wrapper)
        self._wrap(agent_module._PriorityGate, "_enqueue_and_wait", queue_wrapper)
        return self

    def restore(self):
        for cls, name, original in reversed(self.patches):
            setattr(cls, name, original)
        self.patches.clear()

    def publish_noop_action(self, session, result):
        """Publish REAL action outcomes to ToolAgent's v9.1 registry.

        Competition hardening:
        the same analyzer/diag slot may be reused by later hidden games.
        A new game session therefore starts a fresh registry entry and clears
        this analyzer's per-level trigger memory. This prevents cross-game
        leakage while leaving the tested v9.1 thresholds unchanged.
        """
        if not isinstance(result, dict):
            return
        if str(os.environ.get("ARC3_NOOP_RECOVERY_V91", "0")).strip().lower() not in {
            "1", "true", "yes", "on"
        }:
            return

        analyzer = getattr(session, "analyzer", None)
        diag_name = str(getattr(analyzer, "_diag_name", "") or "")
        if not diag_name:
            return

        from inference.agent import tool_agent as _tool_agent_module

        registry = getattr(_tool_agent_module, "_ARC3_NOOP_ACTION_REGISTRY", None)
        if not isinstance(registry, dict):
            registry = {}
            setattr(_tool_agent_module, "_ARC3_NOOP_ACTION_REGISTRY", registry)

        game = getattr(session, "game", None)
        run = getattr(game, "game_run", None)
        game_id = str(getattr(run, "game_id", getattr(game, "env_name", "unknown")))
        pass_index = getattr(session, "pass_index", None)
        session_key = f"{game_id}|pass={pass_index}|session={id(session)}"

        level = result.get("level")
        entry = registry.get(diag_name)
        new_session = (
            not isinstance(entry, dict)
            or entry.get("session_key") != session_key
        )
        if new_session or entry.get("level") != level:
            entry = {
                "session_key": session_key,
                "game_id": game_id,
                "level": level,
                "events": [],
            }
            registry[diag_name] = entry

        # Only real solver actions count toward recovery evidence.
        if bool(result.get("automatic")):
            return

        entry["events"].append({
            "action_num": result.get("action_num"),
            "no_op": bool(result.get("no_op")),
            "gameplay_changed": result.get("gameplay_changed"),
            "game_over": bool(result.get("game_over")),
            "level_completed": bool(result.get("level_completed")),
        })

        # Trigger logic only consumes the latest 24; keep a bounded buffer.
        if len(entry["events"]) > 200:
            entry["events"] = entry["events"][-200:]

    def observe_action(self, result):
        row = getattr(self.local, "row", None)
        if row is None or not isinstance(result, dict):
            return
        if row["first_action_seconds"] is None:
            row["first_action_seconds"] = time.monotonic() - row["start"]
        row["counts"]["observed_actions"] += 1
        for key in ("automatic", "no_op", "game_over", "level_completed"):
            row["counts"][key] += int(bool(result.get(key)))

    def observe_tool(self, result):
        row = getattr(self.local, "row", None)
        if row is None:
            return
        row["counts"]["tool_calls"] += 1
        try:
            payload = json.loads(result.content)
        except (TypeError, ValueError, AttributeError):
            row["counts"]["unparsed_tool_payloads"] += 1
            return
        if not isinstance(payload, dict):
            row["counts"]["unparsed_tool_payloads"] += 1
            return
        error = payload.get("error")
        if error:
            row["counts"]["tool_errors"] += 1
            names = re.findall(r"\b([A-Za-z]+(?:Error|Exception))\b", str(error))
            row["errors"][names[-1] if names else "unclassified"] += 1
        notes = str(payload.get("function_retention", ""))
        if "not retained" in notes or "cleared after a retention error" in notes:
            row["counts"]["function_retention_issue_calls"] += 1
            issue_lines = [
                line for line in notes.splitlines()
                if "not retained" in line or "cleared after a retention error" in line
            ]
            row["counts"]["function_retention_rejections"] += max(1, len(issue_lines))

    def finish(self, session, row):
        run = getattr(session.game, "game_run", None)
        state = getattr(session.game, "current_state", None)
        analyzer = getattr(session, "analyzer", None)
        spent = max(0, int(getattr(analyzer, "generated_tokens", 0) or 0)
                    - int(getattr(session, "game_token_baseline", 0) or 0))
        strategy_events = list(getattr(analyzer, "_strategy_checkpoint_events", []) or [])
        record = {
            "profile": self.profile,
            "game_id": str(getattr(run, "game_id", getattr(session.game, "env_name", "unknown"))),
            "game_index": getattr(session, "game_index", None),
            "pass_index": getattr(session, "pass_index", None),
            "levels_completed": getattr(state, "levels_completed", None),
            "total_levels": getattr(session.game, "number_of_levels", None),
            "actions": getattr(session, "action_count", None),
            "generated_tokens": spent,
            "session_seconds": round(time.monotonic() - row["start"], 3),
            "queue_seconds": round(row["queue_seconds"], 3),
            "first_action_seconds": row["first_action_seconds"],
            "run_state_at_solver_exit": str(getattr(run, "state", "unknown")),
            "solver_note": str(getattr(run, "solver_note", "") or ""),
            "exception": row["exception"],
            "counts": dict(row["counts"]), "error_types": dict(row["errors"]),
            "step_verification_prompt_present": "During a multi-action plan, verify important effects" in
                str(getattr(analyzer, "_system_prompt", "")),
            "function_repair_hint_flag": os.environ.get("ARC3_PERSISTENT_FUNCTIONS_REPAIR_HINTS"),
            "persistent_function_scope": os.environ.get("ARC3_PERSISTENT_FUNCTIONS_SCOPE", "levelup"),
            "hypothesis_switch_flag": os.environ.get("ARC3_STAGNATION_HYPOTHESIS_SWITCH", "0"),
            "adaptive_switch_flag": os.environ.get("ARC3_ADAPTIVE_HYPOTHESIS_SWITCH", "0"),
            "cycle_switch_flag": os.environ.get("ARC3_CYCLE_HYPOTHESIS_SWITCH", "0"),
            "strategy_checkpoint_count": len(strategy_events),
            "strategy_checkpoint_events": strategy_events,
        }
        with self.lock:
            self.records.append(record)
            with (self.directory / "telemetry.jsonl").open("a", encoding="utf-8") as file:
                file.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

    def export(self):
        counts = Counter()
        for row in self.records:
            counts.update(row["counts"])
        write_json(self.directory / "telemetry_summary.json", {
            "sessions": len(self.records), "counts": dict(counts),
            "observer_failures": self.observer_failures,
            "queue_seconds": sum(row["queue_seconds"] for row in self.records),
            "exceptions": sum(row["exception"] is not None for row in self.records),
            "metric_notes": {
                "no_op": "Existing harness interior-grid heuristic; not proof that an action was useless.",
                "queue_seconds": "Time inside the harness admission queue, not server-side queue time.",
                "session_seconds": "Includes admission waiting; not exclusive GPU time.",
                "run_state_at_solver_exit": "Framework may finalize state after this observer; use finished log for final state.",
                "tool_errors": "Top-level JSON error fields only; consult unparsed_tool_payloads and transcripts.",
            },
        })


FINISHED_RE = re.compile(
    r"\[finished\]\s+(\S+)\s+state=(\S+)\s+level=(\d+)/(\d+)\s+"
    r"score=([\d.]+)\s+actions=(\d+)\s+tokens=(\d+)"
)


def parse_finished_log(path):
    """Parse every [finished] record, even when concurrent prints share a line.

    The old regex was anchored with ^, so if two worker threads printed their
    [finished] records back-to-back on one physical line, the second game was
    silently omitted and the whole arm could be marked incomplete.
    """
    content = Path(path).read_text(encoding="utf-8", errors="replace")
    return [{
        "game_id": match[1], "state": match[2],
        "levels_completed": int(match[3]), "total_levels": int(match[4]),
        "score": float(match[5]), "actions": int(match[6]),
        "log_tokens": int(match[7]),
    } for match in FINISHED_RE.finditer(content)]


def load_telemetry_rows(directory):
    path = Path(directory) / "telemetry.jsonl"
    rows = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            item = json.loads(line)
        except (TypeError, ValueError):
            continue
        if isinstance(item, dict):
            rows.append(item)
    return rows


def _ratio(numerator, denominator):
    try:
        denominator = float(denominator)
        return float(numerator) / denominator if denominator else None
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def export_comparison(root, completed_arms):
    root = Path(root)
    rows, per_game = [], []
    arm_games = {}
    for arm in completed_arms:
        directory = Path(arm["directory"])
        finished = parse_finished_log(directory / "run.log")
        telemetry_rows = load_telemetry_rows(directory)
        telemetry_by_game = {str(item.get("game_id")): item for item in telemetry_rows}
        telemetry_path = directory / "telemetry_summary.json"
        telemetry = json.loads(telemetry_path.read_text()) if telemetry_path.exists() else {}
        complete = len(finished) == int(arm["expected_runs"]) and arm["returncode"] == 0

        enriched = []
        for item in finished:
            record = dict(item)
            observed = telemetry_by_game.get(record["game_id"], {})
            counts = observed.get("counts", {}) if isinstance(observed, dict) else {}
            # Analyzer-generated-token telemetry is the preferred efficiency count.
            # Keep the framework log count too so disagreements are visible rather than hidden.
            telemetry_tokens = observed.get("generated_tokens") if isinstance(observed, dict) else None
            record.update({
                "tokens": telemetry_tokens if telemetry_tokens is not None else record["log_tokens"],
                "telemetry_tokens": telemetry_tokens,
                "token_count_delta": (
                    int(telemetry_tokens) - int(record["log_tokens"])
                    if telemetry_tokens is not None else None
                ),
                "tool_calls": counts.get("tool_calls"),
                "tool_errors": counts.get("tool_errors"),
                "function_retention_issue_calls": counts.get("function_retention_issue_calls"),
                "function_retention_rejections": counts.get("function_retention_rejections"),
                "no_op": counts.get("no_op"),
                "automatic": counts.get("automatic"),
                "persistent_function_scope": observed.get("persistent_function_scope"),
                "strategy_checkpoint_count": observed.get("strategy_checkpoint_count", 0),
                "strategy_checkpoint_events": json.dumps(observed.get("strategy_checkpoint_events", []), ensure_ascii=False, sort_keys=True),
                "error_types": json.dumps(observed.get("error_types", {}), ensure_ascii=False, sort_keys=True),
            })
            record["tool_error_rate"] = _ratio(record.get("tool_errors"), record.get("tool_calls"))
            record["noop_rate"] = _ratio(record.get("no_op"), record.get("actions"))
            enriched.append(record)
            per_game.append({"arm": directory.name, "profile": arm["profile"], **record})
        arm_games[directory.name] = {item["game_id"]: item for item in enriched}

        scores = [x["score"] for x in enriched]
        completed_levels = sum(x["levels_completed"] for x in enriched)
        actions = sum(x["actions"] for x in enriched)
        telemetry_tokens = [x["tokens"] for x in enriched if x.get("tokens") is not None]
        counts = telemetry.get("counts", {}) if telemetry else {}
        rows.append({
            "arm": directory.name, "profile": arm["profile"], "complete": complete,
            "returncode": arm["returncode"], "finished_runs": len(enriched),
            "expected_runs": arm["expected_runs"],
            "mean_score": sum(scores) / len(scores) if complete and scores else None,
            "median_score": statistics.median(scores) if complete and scores else None,
            "completed_levels": completed_levels,
            "wins": sum(x["state"] == "won" for x in enriched),
            "actions": actions,
            "actions_per_completed_level": _ratio(actions, completed_levels),
            "tokens": sum(telemetry_tokens) if telemetry_tokens else None,
            "log_tokens": sum(x["log_tokens"] for x in enriched),
            "token_count_delta": sum(
                x.get("token_count_delta") or 0 for x in enriched
            ),
            "wall_seconds": arm["wall_seconds"],
            "tool_calls": counts.get("tool_calls", 0) if telemetry else None,
            "tool_errors": counts.get("tool_errors", 0) if telemetry else None,
            "tool_error_rate": _ratio(counts.get("tool_errors", 0), counts.get("tool_calls", 0)) if telemetry else None,
            "no_op": counts.get("no_op", 0) if telemetry else None,
            "noop_rate": _ratio(counts.get("no_op", 0), actions) if telemetry else None,
            "function_retention_issue_calls": counts.get("function_retention_issue_calls", 0) if telemetry else None,
            "function_retention_rejections": counts.get("function_retention_rejections", 0) if telemetry else None,
            "strategy_checkpoint_count": sum(int(x.get("strategy_checkpoint_count") or 0) for x in enriched),
            "queue_seconds": telemetry.get("queue_seconds"),
            "observer_failures": telemetry.get("observer_failures"),
            "cache_flush_confirmed": arm["cache_flush_confirmed"],
        })

    for filename, values in (("comparison.csv", rows), ("per_game.csv", per_game)):
        if values or filename == "per_game.csv":
            with (root / filename).open("w", newline="", encoding="utf-8-sig") as file:
                fields = list(values[0]) if values else [
                    "arm", "profile", "game_id", "state", "levels_completed",
                    "total_levels", "score", "actions", "tokens", "log_tokens"]
                writer = csv.DictWriter(file, fieldnames=fields)
                writer.writeheader()
                writer.writerows(values)

    # Repeat-aware paired comparison.
    # Canonical A/B order comes from campaign.json, not execution order,
    # so reversed repeat-2 arms still compute the same B-A direction.
    pair_summary = None
    paired_rows = []
    complete_rows = [row for row in rows if row.get("complete")]

    campaign_path = root / "campaign.json"
    campaign_meta = json.loads(campaign_path.read_text(encoding="utf-8")) if campaign_path.exists() else {}
    requested_profiles = list(campaign_meta.get("profiles") or [])
    if len(requested_profiles) == 2:
        profile_a, profile_b = requested_profiles
        by_repeat = {}
        repeat_re = re.compile(r"^r(\d+)_")

        for row in complete_rows:
            match = repeat_re.match(str(row.get("arm") or ""))
            if not match:
                continue
            repeat_id = int(match.group(1))
            by_repeat.setdefault(repeat_id, {})[row["profile"]] = row

        per_repeat = []
        for repeat_id in sorted(by_repeat):
            group = by_repeat[repeat_id]
            if profile_a not in group or profile_b not in group:
                continue

            left, right = group[profile_a], group[profile_b]
            left_games = arm_games.get(left["arm"], {})
            right_games = arm_games.get(right["arm"], {})
            common = sorted(set(left_games) & set(right_games))
            repeat_deltas = []

            for game_id in common:
                a, b = left_games[game_id], right_games[game_id]
                delta = b["score"] - a["score"]
                repeat_deltas.append(delta)
                paired_rows.append({
                    "repeat": repeat_id,
                    "game_id": game_id,
                    "profile_a": profile_a,
                    "profile_b": profile_b,
                    "score_a": a["score"], "score_b": b["score"],
                    "score_delta_b_minus_a": delta,
                    "levels_a": a["levels_completed"], "levels_b": b["levels_completed"],
                    "levels_delta_b_minus_a": b["levels_completed"] - a["levels_completed"],
                    "actions_a": a["actions"], "actions_b": b["actions"],
                    "actions_delta_b_minus_a": b["actions"] - a["actions"],
                    "tool_errors_a": a.get("tool_errors"), "tool_errors_b": b.get("tool_errors"),
                    "no_op_a": a.get("no_op"), "no_op_b": b.get("no_op"),
                    "strategy_checkpoints_a": a.get("strategy_checkpoint_count"),
                    "strategy_checkpoints_b": b.get("strategy_checkpoint_count"),
                })

            if repeat_deltas:
                per_repeat.append({
                    "repeat": repeat_id,
                    "paired_games": len(repeat_deltas),
                    "mean_score_delta_b_minus_a": sum(repeat_deltas) / len(repeat_deltas),
                    "median_score_delta_b_minus_a": statistics.median(repeat_deltas),
                    "games_improved": sum(delta > 0 for delta in repeat_deltas),
                    "games_worse": sum(delta < 0 for delta in repeat_deltas),
                    "games_tied": sum(delta == 0 for delta in repeat_deltas),
                })

        if paired_rows:
            deltas = [row["score_delta_b_minus_a"] for row in paired_rows]

            # Exclude all repeats of one game at a time. This is more meaningful
            # than excluding one stochastic run when DEV_REPEATS > 1.
            leave_one_game_out = []
            for game_id in sorted({row["game_id"] for row in paired_rows}):
                others = [
                    row["score_delta_b_minus_a"]
                    for row in paired_rows
                    if row["game_id"] != game_id
                ]
                if others:
                    leave_one_game_out.append({
                        "excluded_game": game_id,
                        "mean_delta_b_minus_a": sum(others) / len(others),
                    })

            pair_summary = {
                "profile_a": profile_a,
                "profile_b": profile_b,
                "paired_repeats": len(per_repeat),
                "paired_game_runs": len(paired_rows),
                "unique_games": len({row["game_id"] for row in paired_rows}),
                "mean_score_delta_b_minus_a": sum(deltas) / len(deltas),
                "median_score_delta_b_minus_a": statistics.median(deltas),
                "games_improved": sum(delta > 0 for delta in deltas),
                "games_worse": sum(delta < 0 for delta in deltas),
                "games_tied": sum(delta == 0 for delta in deltas),
                "per_repeat": per_repeat,
                "leave_one_game_out": leave_one_game_out,
            }

            with (root / "paired_delta.csv").open("w", newline="", encoding="utf-8-sig") as file:
                writer = csv.DictWriter(file, fieldnames=list(paired_rows[0]))
                writer.writeheader()
                writer.writerows(paired_rows)
            write_json(root / "pairwise_summary.json", pair_summary)

    write_json(root / "comparison.json", {
        "arms": rows,
        "pairwise": pair_summary,
        "interpretation": (
            "Exploratory public-game screening; no automatic winner or submission. "
            "Inspect paired_delta.csv and leave-one-out sensitivity before attributing a mean-score gain."
        ),
    })
    for row in rows:
        score = "INCOMPLETE" if row["mean_score"] is None else f"{row['mean_score']:.2f}"
        print(
            f"[comparison] {row['arm']}: score={score}, levels={row['completed_levels']}, "
            f"actions={row['actions']}, tool_errors={row['tool_errors']}, "
            f"retention_issues={row['function_retention_issue_calls']}", flush=True
        )
    if pair_summary:
        print(
            f"[paired] {pair_summary['profile_b']} - {pair_summary['profile_a']}: "
            f"mean={pair_summary['mean_score_delta_b_minus_a']:.2f}, "
            f"median={pair_summary['median_score_delta_b_minus_a']:.2f}, "
            f"W/L/T={pair_summary['games_improved']}/{pair_summary['games_worse']}/{pair_summary['games_tied']}",
            flush=True,
        )
    return rows


def wait_for_model(base_url, timeout_seconds=900):
    root = base_url.rstrip("/").removesuffix("/v1")
    deadline, last_report = time.monotonic() + timeout_seconds, 0.0
    while time.monotonic() < deadline:
        try:
            with urlopen(root + "/health", timeout=5) as response:
                if response.status == 200:
                    return
        except (OSError, URLError):
            pass
        if time.monotonic() - last_report >= 30:
            print("Waiting for the shared model before starting the development timer...", flush=True)
            last_report = time.monotonic()
        time.sleep(2)
    raise RuntimeError("Model not healthy; no development arm was started")


def flush_model_cache(base_url, timeout_seconds=90):
    """Only used between sequential development arms, never in competition."""
    root = base_url.rstrip("/").removesuffix("/v1")
    deadline = time.monotonic() + timeout_seconds
    last_error = ""
    while time.monotonic() < deadline:
        try:
            request = Request(root + "/flush_cache", method="POST")
            with urlopen(request, timeout=15) as response:
                body = response.read().decode(errors="replace")
                if response.status == 200 and "cache flushed" in body.lower():
                    return body.strip()
                last_error = f"Unconfirmed response: {response.status} {body[:160]}"
        except HTTPError as exc:
            last_error = f"HTTP {exc.code}"
            if exc.code in (404, 405, 401, 403):
                raise RuntimeError(f"Cannot establish equal cache conditions: {last_error}") from exc
        except (OSError, URLError) as exc:
            last_error = type(exc).__name__
        time.sleep(2)
    raise RuntimeError(f"Model cache was not confirmed empty: {last_error}. No next arm was started.")


async def development_worker(config):
    if competition_rerun():
        raise RuntimeError("Development worker cannot run in a competition rerun")
    validate_development_config(config)
    # Must precede any pickle restore/import of inference modules.
    os.environ.update(PROFILES[config["profile"]])
    os.environ.update({
        "TAAF_RUN_AS_SUBMISSION": "0",
        "TAAF_MINIMAL_DIAGNOSTICS": "0",
        "RECORDINGS_DIR": str(Path(config["output_dir"]) / "recordings"),
        "ARC3_MAX_ACTIVE_STREAMS": str(config["active_streams"]),
        # run_campaign already waits for /health before starting an arm.
        # Disable the submission-only warmup RESET here so paired dev arms do not
        # spend an automatic action on whichever concurrent game happens to start first.
        "ARC3_WARMUP_ACTION_GAMES": "0",
    })
    bundle, output = Path(config["bundle_dir"]), Path(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    for repo in sorted((bundle / "src").iterdir(), reverse=True):
        for candidate in (repo / "src", repo):
            if candidate.is_dir():
                sys.path.insert(0, str(candidate))
    import arc_agi
    import taaf.game_api
    with (bundle / "deploy_target.pkl").open("rb") as file:
        target = pickle.load(file)
    target.actual_run_as_submission = False
    target.is_competition_rerun = False
    with (bundle / "benchmark_initial.pkl").open("rb") as file:
        benchmark = pickle.load(file)
    benchmark.job_dir = output
    spec = taaf.game_api.ArcadeSpec(operation_mode=arc_agi.OperationMode.OFFLINE,
                                   environments_dir=config["environment_dir"])
    arcade = arc_agi.Arcade(operation_mode=arc_agi.OperationMode.OFFLINE,
                            environments_dir=config["environment_dir"])
    game_ids = select_game_ids([item.game_id for item in arcade.available_environments], config["games"])
    benchmark.games = [taaf.game_api.GameAPI(env_name=game_id, arcade_spec=spec) for game_id in game_ids]
    soft_end = apply_development_budget(benchmark, config, len(game_ids))
    active = os.environ["ARC3_MAX_ACTIVE_STREAMS"]
    write_json(output / "effective_config.json", {
        **config, "resolved_game_ids": game_ids, "active_streams_effective": active,
        "solver": repr(benchmark.solver), "profile_flags": PROFILES[config["profile"]],
        "warmup_action_games": os.environ.get("ARC3_WARMUP_ACTION_GAMES"),
        "soft_end": soft_end.isoformat(), "created_utc": datetime.now(timezone.utc).isoformat(),
        "sampling_note": "Temperature is unchanged. GPU decoding is stochastic; this is not a deterministic paired-seed test.",
    })
    print(f"[experiment] profile={config['profile']} games={game_ids} "
          f"passes={benchmark.n_passes} active={active} budget={config['minutes']}min", flush=True)
    print(f"[experiment] flags={PROFILES[config['profile']]}", flush=True)
    observer = Telemetry(output, config["profile"]).install()
    try:
        await benchmark.run(soft_end_time=soft_end, runtime_environment=target, minimal_diagnostics=False)
    finally:
        observer.restore()
        observer._safe(observer.export)


async def run_campaign(*, helper_path, bundle_dir, working_dir, environment_dir,
                       base_url, mode="compare", profiles=("repair_only", "cycle_switch"),
                       games=("tn36", "m0r0", "lp85", "lf52"), minutes=20,
                       repeats=1, active_streams=4, save_request_logs=False):
    if competition_rerun():
        raise RuntimeError("Development campaign is disabled in competition reruns")
    if mode not in {"smoke", "compare", "confirm"}:
        raise ValueError("DEV_MODE must be smoke, compare, or confirm")
    profiles, games = tuple(profiles), tuple(games)
    if not profiles or len(set(profiles)) != len(profiles):
        raise ValueError("Use distinct profiles")
    if len(profiles) > len(PROFILES):
        raise ValueError("Too many profiles requested")
    unknown = [profile for profile in profiles if profile not in PROFILES]
    if unknown:
        raise ValueError(f"Unknown profiles: {unknown}")
    if not games or not 1 <= int(repeats) <= 4:
        raise ValueError("Choose development games and 1 to 4 repeats")
    minutes = 5 if mode == "smoke" else minutes
    selected = games[:2] if mode == "smoke" else games
    campaign_name = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + f"_{time.time_ns() % 1000000:06d}"
    root = Path(working_dir) / "experiments" / campaign_name
    root.mkdir(parents=True, exist_ok=False)
    shared = dict(bundle_dir=str(bundle_dir), environment_dir=str(environment_dir), games=selected,
                  minutes=minutes, passes=1, active_streams=active_streams,
                  save_request_logs=save_request_logs)
    for profile in profiles:
        validate_development_config({**shared, "profile": profile})
    write_json(root / "campaign.json", {**shared, "mode": mode, "profiles": profiles,
                                       "repeats": repeats, "profile_definitions": PROFILES})
    print(f"[campaign] {root}\nNominal play budget: {float(minutes) * len(profiles) * int(repeats):.0f} min "
          "plus setup/teardown. Model weights are loaded once.", flush=True)
    await asyncio.to_thread(wait_for_model, base_url)
    completed = []
    for repeat in range(int(repeats)):
        # Alternate order on repeats to reduce (not eliminate) warmup/order effects.
        ordered = profiles if repeat % 2 == 0 else profiles[::-1]
        for position, profile in enumerate(ordered):
            arm = root / f"r{repeat + 1:02d}_{position + 1}_{profile}"
            arm.mkdir()
            flush_message = await asyncio.to_thread(flush_model_cache, base_url)
            config = {**shared, "profile": profile, "output_dir": str(arm)}
            config_path = arm / "config.json"
            write_json(config_path, config)
            child_env = dict(os.environ)
            child_env.update(PROFILES[profile])
            child_env["PYTHONUNBUFFERED"] = "1"
            child_env["ARC3_EXPERIMENT_ARM"] = arm.name
            start = time.monotonic()
            print(f"[campaign] start {arm.name}; cache flush confirmed", flush=True)
            timed_out = False
            with (arm / "run.log").open("wb") as log:
                process = await asyncio.create_subprocess_exec(
                    sys.executable, str(helper_path), "--worker", str(config_path),
                    env=child_env, stdout=log, stderr=subprocess.STDOUT)
                # Budget starts inside worker; allow bounded import and graceful teardown.
                hard_deadline = time.monotonic() + float(minutes) * 60 + 300
                try:
                    while True:
                        try:
                            await asyncio.wait_for(process.wait(), timeout=30)
                            break
                        except asyncio.TimeoutError:
                            if time.monotonic() >= hard_deadline:
                                timed_out = True
                                process.terminate()
                                try:
                                    await asyncio.wait_for(process.wait(), timeout=15)
                                except asyncio.TimeoutError:
                                    process.kill()
                                    await process.wait()
                                break
                            print(f"[campaign] {arm.name}: {(time.monotonic() - start)/60:.1f} min; "
                                  f"live log: {arm / 'run.log'}", flush=True)
                except BaseException:
                    if process.returncode is None:
                        process.terminate()
                        try:
                            await asyncio.wait_for(process.wait(), timeout=15)
                        except asyncio.TimeoutError:
                            process.kill()
                            await process.wait()
                    raise
            result = {"directory": str(arm), "profile": profile,
                      "expected_runs": len(selected), "returncode": process.returncode,
                      "wall_seconds": round(time.monotonic() - start, 3),
                      "cache_flush_confirmed": bool(flush_message), "hard_timeout": timed_out}
            completed.append(result)
            write_json(arm / "process_result.json", result)
            export_comparison(root, completed)
            if process.returncode != 0 or timed_out:
                tail = (arm / "run.log").read_text(errors="replace")[-6000:]
                print(tail, flush=True)
                raise RuntimeError(f"{arm.name} did not finish cleanly; remaining arms were not started")
    print(f"[campaign] Finished. Review {root / 'comparison.csv'} and per-game transcripts. "
          "No winner was selected and no competition submission was made.", flush=True)
    return root


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", type=Path, required=True)
    arguments = parser.parse_args()
    asyncio.run(development_worker(json.loads(arguments.worker.read_text(encoding="utf-8"))))

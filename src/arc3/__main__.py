"""CLI entry: python -m arc3 --config configs/baseline.json."""
import argparse
import asyncio
import json
from .config import Settings

async def run(settings):
    from .environment import prepare
    from .runtime import install
    from .precache import prepare_precache
    from .serving import launch
    from .benchmark import execute
    from .monitor import start_monitor, _sysline
    settings.check_inputs()
    ctx = prepare(settings)
    precache = prepare_precache(ctx)
    install(ctx)
    server = None
    monitor = None
    try:
        server = launch(ctx, precache)
        if not ctx.TRUE_SUBMISSION:
            monitor = start_monitor(2)
            print(_sysline())
        await execute(ctx)
    finally:
        if monitor is not None:
            monitor.set()
        if server is not None and server.poll() is None:
            import os, signal, subprocess
            os.killpg(server.pid, signal.SIGTERM)
            try:
                server.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(server.pid, signal.SIGKILL)
                server.wait()

def main():
    parser = argparse.ArgumentParser(description='ARC3 Agent inference system')
    parser.add_argument('--config', default='configs/baseline.json')
    parser.add_argument('--check', action='store_true', help='Validate config and print plan; no model loading')
    args = parser.parse_args()
    settings = Settings.load(args.config)
    if args.check:
        print(json.dumps({'status': 'config-valid', 'run': settings.run,
                          'inference': settings.inference}, ensure_ascii=False, indent=2))
        return
    asyncio.run(run(settings))

if __name__ == '__main__':
    main()

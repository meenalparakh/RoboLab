# Evaluation fixes from RoboLab Verified

The source fixes are from [Manda Robotics/RoboLab-Verified](https://github.com/Manda-Robotics/RoboLab-Verified).
Their original authorship and source commit IDs are retained in Git. This port
includes the evaluation, recording, dashboard, scene and tooling fixes. The source
fork's additional robots and policy backends are excluded.

## Validate without a simulator

Run from the RoboLab checkout, using its configured Python environment:

```bash
git lfs pull
python -m pytest offline_tests -q
python -m pytest --noconftest tests/test_trajectory_metrics.py tests/test_lfs_pointers.py -q
python scripts/audit_task_definitions.py
node --check dashboard/static/app.js
```

The offline suite needs pytest, torch, numpy, scipy, h5py, pyyaml, usd-core,
fastapi, httpx and jinja2. The CI workflow installs these without Isaac Sim.
An audit exit code of zero means no asserted conflicts; inspect its informational
notes and cases it could not assert separately.

## Evaluate and inspect recordings

Use the Isaac stack and policy server configured for the experiment. These examples
use an existing upstream policy backend:

```bash
uv run --no-sync python policies/pi0_family/run.py --task BananaInBowlTask --headless --friction upstream
uv run --no-sync python policies/pi0_family/run.py --task BananaInBowlTask --headless --friction 0.5
python scripts/verify_patches.py output/patched_run --baseline output/baseline_run --summary
python scripts/reflag.py output/patched_run --json output/reflagged.json
python scripts/annotate_phases.py output/patched_run --summary --check-events --check-replay
python scripts/flag_tows.py output/patched_run --csv output/tows.csv
```

Substitute actual run directories. The default friction remains the authored
upstream material; a coefficient is an explicit experiment override. Consult
[the migration guide](../verified/migration.md) before comparing results: the
scoring and event definitions change.

## Bound a replay

```bash
uv run --no-sync python examples/run_recorded.py --task BananaInBowlTask --recorded-data-folder output/patched_run --file run_0.hdf5 --episode 0 --headless --max-steps 180 --no-video --output-dir output/replay_check
```

Use [the script index](../../scripts/README.md) and each script's `--help` for the
remaining audit and annotation tools. Runtime evidence in the imported Verified
register describes the source fork. It is not a fresh GPU validation of this port.

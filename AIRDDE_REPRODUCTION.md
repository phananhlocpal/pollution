# AirDDE reproduction

The official repository is pinned as the `third_party/airdde` submodule. Its public
release does not include `KnowAir.npy` or pretrained checkpoints.

Prepare the shared dataset from the repository root:

```powershell
.\.venv\Scripts\python.exe scripts\prepare_airdde.py
```

Train the exact released configuration (MAE loss, patience 3):

```powershell
.\.venv\Scripts\python.exe scripts\run_airdde.py train
```

The wrapper retains checkpoints by seed under `artifacts/airdde/seed_<seed>`.
The following multi-seed run is optional and only needed for a release-vs-release
fairness study; it is not required to compare against the published paper table:

```powershell
foreach ($seed in 42,43,44) {
  .\.venv\Scripts\python.exe scripts\run_airdde.py train --random_seed $seed
}
```

The paper configuration (reported as Huber loss and early-stopping tolerance 10)
must be recorded as a separate experiment; it must not silently replace the release
configuration.

The wrapper exposes an explicitly approximate version without editing the pinned
checkout:

```powershell
.\.venv\Scripts\python.exe scripts\run_airdde.py train --random_seed 42 --paper-style
```

This redirects the release trainer's unsupported criterion fallback to SmoothL1
and uses patience 10. It is not an exact reproduction of the paper's Bayesian HPO.

Run the official test exporter after training:

```powershell
.\.venv\Scripts\python.exe scripts\run_airdde.py eval
.\.venv\Scripts\python.exe scripts\import_airdde_predictions.py
.\.venv\Scripts\python.exe -m benchmarking.evaluator artifacts\predictions\airdde_repro_test
```

`scripts/run_airdde.py` leaves the pinned checkout untouched. It only removes the
obsolete no-op `verbose` argument passed to `ReduceLROnPlateau`, which PyTorch 2.8
no longer accepts. NumPy is constrained to 1.26 because the release uses `np.Inf`.

For a paired comparison, export `common_local` on the same test split only after the
architecture is frozen, then pass its bundle via `--compare`.

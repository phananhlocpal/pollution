# External replication boundary

Only KnowAir and China-AQI remain in scope. US-PM was dropped because a verifiable
public source could not be established.

## KnowAir / AirDDE

AirDDE is represented only by the published scalar references recorded in
`paper/RESULTS.json`. The local AirDDE repository, checkpoints, logs and imported
predictions were removed. The retained model consumes realized target-period core
meteorology, so this is not an identical-input comparison and must not be framed
as paired statistical superiority.

## China-AQI

The corrected protocol is 96 historical hours to 24 forecast hours on 209
stations. The local GAGNN data can be prepared and audited through
`scripts/prepare_gagnn.py` and `scripts/audit_gagnn_reconstruction.py`.

The corrected test split remains unopened. Historical 24-to-6 results are not
part of the paper comparison. Protocol evidence is retained in
`paper/artifacts/external_protocols.json` and
`paper/artifacts/china_aqi_reconstruction.json`.

# Data safety

LLM Maker stores training metadata in SQLite and checkpoints on disk. This page
describes what protects that data, where it lives, and how to recover.

## Where data lives

| Path | Contents |
| --- | --- |
| `studio.db` (desktop project root) | Desktop runs, metrics, alerts, settings, checkpoints, datasets, workspaces |
| `data/rocanon_studio.db` | Database used by the browser/server workflow (kept separate by design) |
| `.secrets.key` next to the database | Encrypted data key for notification secrets; never commit or share it |
| `checkpoints_v2/`, `workspaces/`, `data/uploads/` | Model checkpoints, workspace files, uploaded datasets |
| `data/exports/` | Temporary workspace archives; deleted after the download finishes |
| `backups/` | Automatic pre-migration database snapshots |
| `tauri_debug.log` | Desktop startup/worker log (rotated) |

SQLite runs in WAL mode. Do not copy a database while the app is running unless
you use the backup mechanism below; copy the `-wal` and `-shm` files too.

## Integrity checks

Every open runs `PRAGMA quick_check` before migrations. A database that fails the
check raises `DatabaseIntegrityError` (Python) or refuses to start (Rust) with the
database path in the message; the app does not continue on a corrupt file. Keep the
pre-migration snapshots: they are the recovery point.

## Schema migrations and backups

The schema is defined once, in `src/db/migrations/*.sql`. Both the Python
engine and the Rust shell read the same files, apply pending versions in order,
and record them in `schema_migrations`. This keeps the two layers from drifting.

Before applying any pending migration to a non-empty database, the app writes a
consistent snapshot with SQLite's online backup API to:

```
backups/<database>-<timestamp>.db
```

The Python server uses `YYYYMMDD-HHMMSS`; the desktop shell uses Unix seconds.

The newest 10 snapshots are kept. A fresh database needs no snapshot. If a
snapshot cannot be written for a populated database, the upgrade is refused and
startup fails with an explanatory error. If a migration fails, the app
exits/raises instead of continuing with a half-upgraded schema; the snapshot is
the recovery point.

### Restoring a snapshot

1. Close the app and make sure no worker is running.
2. Copy the chosen `backups/<database>-*.db` over the live database (`studio.db`
   on desktop, `data/rocanon_studio.db` for the server).
3. Delete `studio.db-wal` and `studio.db-shm` if present.
4. Start the app and check the run list.

Restoring rolls back schema and data to the snapshot; checkpoints referenced by
the snapshot remain on disk.

## Crash recovery

On startup the desktop shell marks runs left in `STARTING`, `RUNNING`,
`PAUSING`, `PAUSED`, or `STOPPING` as `STOPPED`, and logs how many were
recovered. This prevents a killed session from leaving phantom active runs.
Metrics written before the crash remain; the last in-flight step may be missing.

## Workspace archives

Settings → Recovery & Diagnostics can export the active workspace as a ZIP. The
archive contains a versioned `manifest.json`, the workspace/run/metric/checkpoint/
dataset rows, and only the files those rows reference. Every file carries its
SHA-256 checksum and size.

Import validates the manifest schema and checksums, rejects absolute paths and
`..` entries, and always creates a new workspace id. It never overwrites an
existing workspace. Archives are limited to 4 GiB and 20,000 entries.

## Safe mode

Set `LLMMAKER_SAFE_MODE=1` before starting the app or server to disable plugin
discovery and plugin changes. No sample plugins are generated and the enabled
plugin set is left untouched. Training and data access still work. The Settings
panel also exposes a persisted safe-mode switch for the next start; the
environment variable wins and the API returns 409 while it is set.

## Secrets

`discord_webhook_url` and `telegram_bot_token` are encrypted with AES-256-GCM
before they are written to SQLite. The Python control plane and the Rust desktop
shell share the same ciphertext format and the same key file, so either process
can open the database. The data key is protected with Windows DPAPI
(`CryptProtectData`) and stored next to the database (`data/.secrets.key` for the
Python data DB, `.secrets.key` in the desktop project root); on other platforms
the key file is written with owner-only permissions (`0600`).

The HTTP API returns only `<key>_configured` booleans and the sentinel
`__LLMMAKER_SECRET_SET__`. Saving the sentinel keeps the stored value, an empty
string clears it, and omitting the field leaves it unchanged. Plaintext values
from older databases are encrypted the first time the database is opened.

**Back up `.secrets.key` with the database.** If the key is lost, stored
credentials cannot be recovered by design. Both the Python and Rust paths refuse
to create a replacement key while encrypted rows exist, and a wrong key returns
an explicit decryption error instead of an empty credential. Treat both files as
sensitive: do not attach them to bug reports or share them publicly.

## What is not automated yet

- No scheduled/offsite backup. Copy the live database, the `.secrets.key` beside
  it (and the matching `-wal`/`-shm` files while closed), `checkpoints_v2/`, and
  `workspaces/`.
- No automatic restore UI; recovery is the file copy above.
- Workspace archives cover one workspace. A full-database archive is not exposed;
  use the snapshot or a closed-app copy for that.

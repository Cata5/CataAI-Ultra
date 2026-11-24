# Staged installer resources

`scripts/build_installer.ps1` replaces `app/` and `python/` here before
`tauri build`. This tracked file keeps Tauri's `resources/**/*` glob valid in
development and CI, where the staged payload is intentionally absent.

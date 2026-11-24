-- Workspace scoping was added after the first public build.
ALTER TABLE runs ADD COLUMN workspace_id TEXT DEFAULT 'default';
ALTER TABLE datasets ADD COLUMN workspace_id TEXT DEFAULT 'default';
ALTER TABLE checkpoints ADD COLUMN workspace_id TEXT DEFAULT 'default';
ALTER TABLE benchmarks ADD COLUMN workspace_id TEXT DEFAULT 'default';
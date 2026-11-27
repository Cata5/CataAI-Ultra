-- Persist the live progress context with every metric point so resumed runs
-- and charts reload epoch, schedule and step latency from SQLite instead of
-- starting from zero. Each statement runs once per database.
ALTER TABLE metrics ADD COLUMN current_epoch INTEGER;
ALTER TABLE metrics ADD COLUMN epoch_pct REAL;
ALTER TABLE metrics ADD COLUMN tokens_per_epoch INTEGER DEFAULT 0;
ALTER TABLE metrics ADD COLUMN tokens_in_epoch INTEGER DEFAULT 0;
ALTER TABLE metrics ADD COLUMN step_time_ms REAL DEFAULT 0;
ALTER TABLE metrics ADD COLUMN duration_mode TEXT;
ALTER TABLE metrics ADD COLUMN target_epochs INTEGER;
ALTER TABLE metrics ADD COLUMN sched_phase TEXT;
ALTER TABLE metrics ADD COLUMN sched_factor REAL;
ALTER TABLE metrics ADD COLUMN eval_interval INTEGER;
ALTER TABLE metrics ADD COLUMN next_val_step INTEGER;

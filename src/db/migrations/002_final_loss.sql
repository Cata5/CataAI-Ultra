-- Add final_loss (terminal loss for completed/stopped runs).
-- Databases created from the version-1 baseline already have the column; the
-- runner treats "duplicate column name" on ALTER statements as already applied.
ALTER TABLE runs ADD COLUMN final_loss REAL;
"use client";

import { useCallback, useEffect, useState } from "react";
import { api, isTauri } from "@/services/api";
import { FlaskConical, Play, RefreshCw, Square, Trash2 } from "lucide-react";

const AXES = [
  "learning_rate", "peak_lr", "muon_lr", "micro_batch_size", "grad_accum_steps",
  "warmup_steps", "warmup_steps_pct", "stable_steps_pct", "weight_decay",
  "dropout", "seq_len", "optimizer", "scheduler",
];

type Sweep = {
  id: string;
  name?: string;
  status?: string;
  best_trial?: { index: number; val_loss: number } | null;
  active_trial?: number | null;
  trials?: Array<{ index: number; status: string; val_loss?: number | null; params?: Record<string, unknown> }>;
};

export function SweepsPanel() {
  const offline = isTauri();
  const [sweeps, setSweeps] = useState<Sweep[]>([]);
  const [networks, setNetworks] = useState<Array<{ id: string; name?: string }>>([]);
  const [datasets, setDatasets] = useState<Array<{ id: string; name?: string }>>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [form, setForm] = useState({
    name: "",
    networkId: "",
    datasetId: "",
    totalSteps: "200",
    axis: "learning_rate",
    values: "1e-4, 3e-4",
  });

  const refresh = useCallback(async () => {
    if (offline) return;
    try {
      const [nextSweeps, nextNetworks, nextDatasets] = await Promise.all([
        api.listSweeps(),
        api.listNetworks(),
        api.listDatasets(),
      ]);
      setSweeps(nextSweeps);
      setNetworks(nextNetworks.map((n: any) => ({ id: n.id, name: n.name || n.id })));
      setDatasets(nextDatasets.map((d: any) => ({ id: d.id || d.dataset_id, name: d.name || d.filename || d.id })));
      setError(null);
    } catch (e: any) {
      setError(e?.message || "Could not load sweeps");
    }
  }, [offline]);

  useEffect(() => {
    refresh();
    if (offline) return;
    const timer = window.setInterval(refresh, 5000);
    return () => window.clearInterval(timer);
  }, [refresh, offline]);

  const createSweep = async () => {
    const values = form.values.split(",").map((value) => value.trim()).filter(Boolean);
    if (!form.networkId || !form.datasetId) {
      setError("Choose a network and a dataset first.");
      return;
    }
    if (values.length === 0) {
      setError("Add at least one axis value.");
      return;
    }
    setLoading(true);
    try {
      const parsed = values.map((value) => {
        const number = Number(value);
        return Number.isFinite(number) ? number : value;
      });
      await api.createSweep({
        name: form.name || undefined,
        base_payload: {
          network_id: form.networkId,
          dataset_id: form.datasetId,
          total_steps: Number(form.totalSteps),
        },
        axes: { [form.axis]: parsed },
      });
      setForm((current) => ({ ...current, name: "" }));
      await refresh();
    } catch (e: any) {
      setError(e?.message || "Could not create sweep");
    } finally {
      setLoading(false);
    }
  };

  const runAction = async (id: string, action: "start" | "stop" | "delete") => {
    if (action === "delete" && !window.confirm("Delete this sweep and its trial history?")) return;
    setLoading(true);
    try {
      if (action === "start") await api.startSweep(id);
      else if (action === "stop") await api.stopSweep(id);
      else await api.deleteSweep(id);
      await refresh();
    } catch (e: any) {
      setError(e?.message || "Sweep action failed");
    } finally {
      setLoading(false);
    }
  };

  if (offline) {
    return (
      <div className="panel p-3 bg-[#202020] border-[#333333] text-[10px] text-[#A3A3A3]">
        Hyperparameter sweeps need the local Python dashboard (port 8081). The desktop shell does not start it yet.
      </div>
    );
  }

  return (
    <div className="panel p-3 bg-[#202020] border-[#333333] space-y-2">
      <div className="flex items-center justify-between gap-2">
        <span className="text-[10px] text-[#D4D4D4] uppercase font-semibold flex items-center gap-1.5">
          <FlaskConical className="w-3 h-3 text-[#A3A3A3]" /> Hyperparameter sweeps
        </span>
        <button type="button" onClick={refresh} className="btn btn-ghost btn-sm" title="Refresh sweeps">
          <RefreshCw className="w-3 h-3" />
        </button>
      </div>

      {error && <div className="p-2 rounded bg-[#2a1f1f] border border-[#5a3333] text-[#fca5a5] text-[10px]">{error}</div>}

      <div className="grid grid-cols-2 lg:grid-cols-6 gap-1.5 text-[10px]">
        <input className="input" placeholder="Sweep name (optional)" value={form.name} onChange={(event) => setForm({ ...form, name: event.target.value })} />
        <select className="input" value={form.networkId} onChange={(event) => setForm({ ...form, networkId: event.target.value })}>
          <option value="">Network...</option>
          {networks.map((network) => <option key={network.id} value={network.id}>{network.name}</option>)}
        </select>
        <select className="input" value={form.datasetId} onChange={(event) => setForm({ ...form, datasetId: event.target.value })}>
          <option value="">Dataset...</option>
          {datasets.map((dataset) => <option key={dataset.id} value={dataset.id}>{dataset.name}</option>)}
        </select>
        <input className="input" type="number" min={1} value={form.totalSteps} onChange={(event) => setForm({ ...form, totalSteps: event.target.value })} />
        <select className="input" value={form.axis} onChange={(event) => setForm({ ...form, axis: event.target.value })}>
          {AXES.map((axis) => <option key={axis} value={axis}>{axis}</option>)}
        </select>
        <input className="input" placeholder="1e-4, 3e-4" value={form.values} onChange={(event) => setForm({ ...form, values: event.target.value })} />
      </div>
      <div className="flex items-center gap-2">
        <button type="button" className="btn btn-primary btn-sm" disabled={loading} onClick={createSweep}>
          <Play className="w-3 h-3" /> Create sweep
        </button>
        <span className="text-[10px] text-[#737373]">One axis, comma-separated values. The first trial starts when the worker is idle.</span>
      </div>

      {sweeps.length === 0 ? (
        <div className="text-[10px] text-[#737373]">No sweeps yet.</div>
      ) : (
        <div className="space-y-1.5">
          {sweeps.map((sweep) => (
            <div key={sweep.id} className="rounded border border-[#333333] bg-[#262626] p-2 space-y-1">
              <div className="flex flex-wrap items-center justify-between gap-2">
                <div className="flex items-center gap-2">
                  <span className="font-medium text-[#F2F2F2]">{sweep.name || sweep.id}</span>
                  <span className="text-[10px] text-[#A3A3A3]">{sweep.status || "unknown"}</span>
                  {typeof sweep.best_trial?.val_loss === "number" && (
                    <span className="text-[10px] text-[#4ade80]">best trial {sweep.best_trial.index}: {sweep.best_trial.val_loss.toFixed(4)}</span>
                  )}
                </div>
                <div className="flex items-center gap-1">
                  {(sweep.status === "interrupted" || sweep.status === "queued") && (
                    <button type="button" className="btn btn-ghost btn-xs" onClick={() => runAction(sweep.id, "start")}>
                      <Play className="w-3 h-3" /> Start
                    </button>
                  )}
                  {(sweep.status === "running" || sweep.status === "queued") && (
                    <button type="button" className="btn btn-ghost btn-xs" onClick={() => runAction(sweep.id, "stop")}>
                      <Square className="w-3 h-3" /> Stop
                    </button>
                  )}
                  {sweep.status !== "running" && (
                    <button type="button" className="btn btn-ghost btn-xs" onClick={() => runAction(sweep.id, "delete")}>
                      <Trash2 className="w-3 h-3" /> Delete
                    </button>
                  )}
                </div>
              </div>
              <div className="flex flex-wrap gap-1">
                {(sweep.trials || []).map((trial) => (
                  <span key={trial.index} className="rounded bg-[#202020] border border-[#333333] px-1.5 py-0.5 text-[9px] text-[#A3A3A3]">
                    #{trial.index} {trial.status}
                    {typeof trial.val_loss === "number" ? ` · ${trial.val_loss.toFixed(4)}` : ""}
                  </span>
                ))}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

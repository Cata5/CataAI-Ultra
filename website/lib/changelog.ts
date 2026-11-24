export const REPO = "https://github.com/Cata5/CataAI-Ultra";

export type ChangeKind = "Feature" | "Fix" | "Docs" | "Chore";

export type ChangeEntry = {
  date: string;
  title: string;
  summary: string;
  kind: ChangeKind;
  sha: string;
};

// Source: the public GitHub history. Release is the current main branch; the
// experimental list contains only commits that are not already on main.
export const releaseChanges: ChangeEntry[] = [
  {
    date: "2026-11-22",
    title: "Minimalist grayscale workstation",
    summary:
      "Replaced the sci-fi interface with the current grayscale design system and the frontend workstation shell.",
    kind: "Feature",
    sha: "1321051",
  },
  {
    date: "2026-11-22",
    title: "Backend stability fixes",
    summary:
      "Corrected backend issues found while wiring the workstation screens to the Python services.",
    kind: "Fix",
    sha: "ba3be13",
  },
  {
    date: "2026-11-21",
    title: "Evaluation laboratory and Network Studio",
    summary:
      "Added the evaluation and architecture-canvas views plus the Python server routes behind them.",
    kind: "Feature",
    sha: "317f51c",
  },
  {
    date: "2026-11-20",
    title: "Streaming, schedule floor, curation and webhooks",
    summary:
      "Added streamed token output, an LR decay floor, dataset curation, and Discord/Telegram webhook alerts.",
    kind: "Feature",
    sha: "1eb5c64",
  },
  {
    date: "2026-11-18",
    title: "FlashAttention-2 kernel",
    summary: "Integrated the FlashAttention-2 path to speed up training on supported GPUs.",
    kind: "Feature",
    sha: "8eb2f55",
  },
  {
    date: "2026-11-17",
    title: "Exact clone URL",
    summary: "Updated the README with the canonical GitHub remote URL.",
    kind: "Docs",
    sha: "b060a6c",
  },
  {
    date: "2026-11-17",
    title: "Launcher encoding cleanup",
    summary: "Cleaned encoding in the PowerShell launcher script.",
    kind: "Chore",
    sha: "d495d20",
  },
  {
    date: "2026-11-17",
    title: "Packaging and cross-platform launchers",
    summary:
      "Packaged the app with setup scripts, requirements, README instructions and platform launchers.",
    kind: "Feature",
    sha: "aff37d4",
  },
  {
    date: "2026-11-17",
    title: "Runs, checkpoints and model registry",
    summary:
      "Added run and checkpoint management views, the model registry, and dataset helper scripts.",
    kind: "Feature",
    sha: "e63ee5e",
  },
  {
    date: "2026-11-16",
    title: "Explicit LM and classification heads",
    summary:
      "Added LM Head and Classification Head palette entries with a dedicated Embed / Head category.",
    kind: "Feature",
    sha: "62cd3fd",
  },
  {
    date: "2026-11-16",
    title: "Studio load, edit, save and switch",
    summary:
      "Enabled full loading, editing, saving and switching of custom model definitions in Network Studio.",
    kind: "Feature",
    sha: "46b9727",
  },
  {
    date: "2026-11-16",
    title: "Premium interface redesign",
    summary:
      "Introduced the earlier sci-fi-inspired neural research interface. Superseded by the grayscale redesign.",
    kind: "Feature",
    sha: "3fb2bc0",
  },
  {
    date: "2026-11-16",
    title: "Architecture presets and scratch flow",
    summary:
      "Added famous architecture presets, a custom scratch flow, and empty-canvas actions.",
    kind: "Feature",
    sha: "16cd91e",
  },
  {
    date: "2026-11-16",
    title: "Universal ML platform",
    summary:
      "Generalized the product from a single model line into the broader LLM Maker platform.",
    kind: "Feature",
    sha: "42ad2cc",
  },
  {
    date: "2026-11-16",
    title: "Backend modules and Next.js application",
    summary:
      "Landed the initial backend source modules, audit reports, and the Next.js frontend application.",
    kind: "Feature",
    sha: "136b223",
  },
  {
    date: "2026-11-15",
    title: "Initial CataAI Ultra framework",
    summary:
      "Started the project with the training web UI, Network Studio and telemetry guards.",
    kind: "Feature",
    sha: "1556586",
  },
];

export const experimentalChanges: ChangeEntry[] = [
  {
    date: "2026-11-24",
    title: "Dashboard logging and hardware monitoring",
    summary:
      "Added dashboard and backend infrastructure for logs, training controls and hardware telemetry.",
    kind: "Feature",
    sha: "62ceec1",
  },
  {
    date: "2026-11-23",
    title: "Tauri desktop shell",
    summary:
      "Added the Rust/Tauri command layer, the shared SQLite database layer, and the desktop views.",
    kind: "Feature",
    sha: "9ebdeb8",
  },
  {
    date: "2026-11-23",
    title: "Desktop workstation documentation",
    summary:
      "Updated the README with the desktop workstation feature tour and UI screenshots.",
    kind: "Docs",
    sha: "8d2c846",
  },
  {
    date: "2026-11-23",
    title: "Training adapters and frontend API surface",
    summary:
      "Added frontend views, the API service, the dataset validator, custom loaders, and DPO/GRPO/ORPO/SimPO adapters.",
    kind: "Feature",
    sha: "25be612",
  },
  {
    date: "2026-11-22",
    title: "Enterprise pillars",
    summary:
      "Added foundation-model loading, multi-GPU support, PEFT/LoRA, and OpenAI-compatible serving.",
    kind: "Feature",
    sha: "b6ee185",
  },
];

export function formatChangeDate(value: string): string {
  const [year, month, day] = value.split("-");
  const months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
  const index = Number(month) - 1;
  if (!year || !months[index] || !day) return value;
  return `${months[index]} ${Number(day)}, ${year}`;
}

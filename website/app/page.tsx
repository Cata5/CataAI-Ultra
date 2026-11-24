import type { ReactNode } from "react";
import { SiteRail } from "@/components/SiteRail";

const REPO = "https://github.com/Cata5/CataAI-Ultra";
const basePath = process.env.NEXT_PUBLIC_BASE_PATH ?? "";

const ledger = [
  { severity: "Critical", found: 4, fixed: 4, open: 0, accepted: 0 },
  { severity: "High", found: 12, fixed: 12, open: 0, accepted: 0 },
  { severity: "Medium", found: 11, fixed: 10, open: 1, accepted: 0 },
  { severity: "Low", found: 7, fixed: 7, open: 0, accepted: 0 },
];

const phases = [
  {
    name: "Adoptable",
    status: "In progress",
    body: "A clean Windows laptop, an NVIDIA driver, a download, a training run — no terminal.",
    shipped: "Bootstrap script, CI, one shared schema, portable data root, setup wizard, bundled-Python staging.",
    open: "Signed installer, auto-update with rollback, clean-machine install proof.",
  },
  {
    name: "Trustworthy",
    status: "Done",
    body: "Losing power mid-run costs at most the current step, and the app explains what happened.",
    shipped: "Pre-migration backups, safe mode, newer-schema refusal, validated checkpoints, encrypted secrets, integrity checks and versioned workspace archives.",
    open: "Offsite backup.",
  },
  {
    name: "Capable",
    status: "Next",
    body: "Scale past a single consumer GPU without pretending the plumbing is the product.",
    shipped: "Single-GPU and DataParallel training, real DDP, checkpoint resume, evaluation hooks, plugin API, GGUF export.",
    open: "Larger dataset pipelines, reproducible evaluation harness, hardware-validated cluster runs.",
  },
  {
    name: "Standard",
    status: "Ongoing",
    body: "Third parties ship plugins and run it in production without asking the maintainer.",
    shipped: "Audit artifacts, versioned schema, documented failure modes, plugin manifests.",
    open: "Release cadence, governance, compatibility policy, community support.",
  },
];

function SectionHead({ id, title, children }: { id: string; title: string; children: ReactNode }) {
  return (
    <div className="section-head">
      <h2 id={id}>{title}</h2>
      <p>{children}</p>
    </div>
  );
}

export default function Home() {
  return (
    <div className="shell">
      <SiteRail page="home" />

      <main>
        <section id="top" className="hero wrap">
          <div className="hero-copy">
            <p className="kicker">Desktop training workstation</p>
            <h1>Training software you can inspect.</h1>
            <p className="lede">
              LLM Maker builds small language models on your own hardware — one GPU or a
              cluster: a visual architecture editor, a real training loop with validation
              and checkpoints, and telemetry you can trace back to the code.
            </p>
            <div className="actions">
              <a className="button" href={REPO}>View repository</a>
              <a className="text-link" href="#audit">Read the audit</a>
            </div>
            <dl className="hero-facts">
              <div><b>58</b>desktop commands</div>
              <div><b>86</b>HTTP routes</div>
              <div><b>354</b>Python tests</div>
            </dl>
          </div>

          <figure className="receipt" aria-label="Benchmark receipt for a 120-step smoke run">
            <div className="receipt-head"><span>RUN RECEIPT</span><span>No. 120</span></div>
            <p className="receipt-title">1.9M-parameter smoke run</p>
            <p className="receipt-sub">RTX 3060 Laptop · 120 steps · FP16</p>
            <div className="receipt-row"><span>median step</span><span>48.36 ms</span></div>
            <div className="receipt-row"><span>throughput</span><span>10,546.6 tok/s</span></div>
            <div className="receipt-row"><span>compute</span><span>0.119 TFLOPs</span></div>
            <div className="receipt-row"><span>VRAM</span><span>0.11 GB</span></div>
            <div className="receipt-row"><span>final loss</span><span>0.1245</span></div>
            <div className="receipt-row"><span>parameters</span><span>1,876,736</span></div>
            <div className="receipt-foot">
              <span>2026-11-24</span>
              <span className="stamp">REPRODUCIBLE</span>
            </div>
          </figure>
        </section>

        <section id="overview" className="wrap">
          <SectionHead id="overview-title" title="A workstation, not a wrapper.">
            Most local training UIs hide the loop behind a form. LLM Maker exposes the
            parts that matter: architecture, schedule, checkpoint state, and the metrics
            the engine actually emitted.
          </SectionHead>

          <ol className="flow">
            <li>
              <span className="step">01</span>
              <h3>Interface</h3>
              <p>Next.js 16 and React 19. 13 pages, 66 typed API methods, Tauri webview or browser.</p>
            </li>
            <li>
              <span className="step">02</span>
              <h3>Shell</h3>
              <p>Rust and Tauri 2 supervise the worker: 58 commands, JSON-lines IPC, process-tree kill.</p>
            </li>
            <li>
              <span className="step">03</span>
              <h3>Engine</h3>
              <p>Python and PyTorch. 86 HTTP routes, 9 worker commands, 13 events, real Muon and AdamW.</p>
            </li>
            <li>
              <span className="step">04</span>
              <h3>State</h3>
              <p>SQLite in WAL mode. Versioned schema, pre-migration backups, resumable optimizer state.</p>
            </li>
          </ol>
        </section>

        <section id="audit" className="wrap">
          <SectionHead id="audit-title" title="Audited before it was polished.">
            Every route, command and worker event was traced, then the critical defects were
            fixed and re-tested. The audit is public in the repository, including what is
            still broken.
          </SectionHead>

          <div className="score">
            <b>7.5</b>
            <span>pre-remediation health score, out of 10</span>
          </div>

          <table className="ledger">
            <caption>34 findings across four severity classes.</caption>
            <thead>
              <tr><th scope="col">Severity</th><th scope="col">Found</th><th scope="col">Fixed</th><th scope="col">Open</th><th scope="col">Accepted</th></tr>
            </thead>
            <tbody>
              {ledger.map((row) => (
                <tr key={row.severity}>
                  <th scope="row">{row.severity}</th>
                  <td className="num">{row.found}</td>
                  <td className="num">{row.fixed}</td>
                  <td className={row.open ? "num open" : "num"}>{row.open}</td>
                  <td className="num">{row.accepted}</td>
                </tr>
              ))}
            </tbody>
            <tfoot>
              <tr><td>Total</td><td className="num">34</td><td className="num">33</td><td className="num open">1</td><td className="num">0</td></tr>
            </tfoot>
          </table>

          <div className="audit-notes">
            <div className="note">
              <strong>What was critical</strong>
              Unsafe checkpoint deserialization, an 800 ms process kill, a stop snapshot that
              captured the wrong state, and resume that silently fell back to random weights.
            </div>
            <div className="note">
              <strong>What stays open</strong>
              One medium item: the missing frontend component test runner. The Cargo target
              configuration is now portable; everything else in the audit is fixed.
            </div>
            <div className="note">
              <strong>How it was checked</strong>
              354 Python tests, a 16-page static frontend build, a production website build,
              Rust <code>cargo check</code>, and a GGUF smoke export. No unexecuted test is claimed.
            </div>
            <div className="note">
              <strong>Since the audit</strong>
              One versioned schema shared by Rust and Python, pre-migration backups with
              retention, safe mode, encrypted notification secrets, startup integrity checks,
              versioned workspace archives, downgrade refusal, a bootstrap script and CI.
              Portable data paths, real DDP, the setup wizard, GGUF export and the plugin API
              landed after the audit was written.
            </div>
          </div>
        </section>

        <section id="roadmap" className="wrap">
          <SectionHead id="roadmap-title" title="From working to standard.">
            The gap between a promising prototype and software a stranger can trust is
            distribution and data safety first, capability second.
          </SectionHead>

          <div className="roadmap">
            {phases.map((phase, index) => (
              <div className="phase" key={phase.name}>
                <div className="phase-num">{String(index + 1).padStart(2, "0")}</div>
                <div>
                  <h3>{phase.name}</h3>
                  <span className="status">{phase.status}</span>
                </div>
                <p>
                  {phase.body} <span className="shipped">Shipped: {phase.shipped}</span> Open: {phase.open}
                </p>
              </div>
            ))}
          </div>
        </section>

        <section id="screens" className="wrap">
          <SectionHead id="screens-title" title="The surface area is the product.">
            Training controller and visual architecture editor, captured from the running
            desktop build.
          </SectionHead>

          <div className="shots">
            <figure className="shot">
              <img
                src={`${basePath}/screenshots/training.png`}
                alt="Training controller with loss, validation, dataset progress, optimizer and schedule panels"
                width={1919}
                height={1007}
                loading="lazy"
              />
              <figcaption>Training controller — schedule, optimizer, batch geometry, live telemetry.</figcaption>
            </figure>
            <div className="shots-pair">
              <figure className="shot">
                <img
                  src={`${basePath}/screenshots/network_studio.png`}
                  alt="Network Studio canvas with transformer blocks, parameter counts and a model inspector"
                  width={1916}
                  height={1005}
                  loading="lazy"
                />
                <figcaption>Network Studio — architecture canvas with parameter accounting.</figcaption>
              </figure>
              <figure className="shot">
                <img
                  src={`${basePath}/screenshots/datasets.png`}
                  alt="Dataset manager showing tokenized corpora and preprocessing status"
                  width={1919}
                  height={1007}
                  loading="lazy"
                />
                <figcaption>Datasets — tokenized corpora, token counts and inspection.</figcaption>
              </figure>
            </div>
          </div>
        </section>

        <footer className="footer wrap">
          <div>
            <p>
              LLM Maker is experimental software. It trains real models on consumer hardware
              and keeps its failures visible. The repository contains the audit, the known
              issues and the verification evidence.
            </p>
            <p className="mono">
              <span>audit/FULL_AUDIT_REPORT.md</span>
              <span>audit/known_issues.md</span>
              <span>audit/pillar_1_2_status.md</span>
            </p>
          </div>
          <a className="button" href={REPO}>View repository</a>
        </footer>
      </main>
    </div>
  );
}
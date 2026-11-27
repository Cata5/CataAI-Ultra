import type { Metadata } from "next";
import type { ReactNode } from "react";
import { SiteRail } from "@/components/SiteRail";

const basePath = process.env.NEXT_PUBLIC_BASE_PATH ?? "";
const REPO = "https://github.com/Cata5/CataAI-Ultra";

export const metadata: Metadata = {
  title: "How to use LLM Maker — from dataset to trained model",
  description:
    "A step-by-step path from a clean dataset to a pretrained base, chat fine-tuning, evaluation and checkpoint resume.",
};

function GuideStep({
  id,
  number,
  label,
  title,
  children,
}: {
  id: string;
  number: string;
  label: string;
  title: string;
  children: ReactNode;
}) {
  return (
    <article className="guide-step" id={id} aria-labelledby={`${id}-title`}>
      <div className="guide-step-rail">
        <span className="guide-step-num">{number}</span>
        <span className="guide-step-label">{label}</span>
      </div>
      <div className="guide-step-body">
        <h2 id={`${id}-title`}>{title}</h2>
        {children}
      </div>
    </article>
  );
}

export default function GuidePage() {
  return (
    <div className="shell">
      <SiteRail page="guide" />

      <main>
        <header className="wrap guide-hero">
          <div className="guide-hero-copy">
            <p className="kicker">User guide</p>
            <h1>From an empty workspace to a model that answers.</h1>
            <p className="lede">
              This is the path the app is built around: bring clean data, pretrain a
              base, fine-tune it for chat, then test the result before you trust it.
            </p>
            <p className="guide-deck">
              The order matters more than the settings. Each step below tells you what
              to press, what to check, and what the app is doing underneath.
            </p>
          </div>

          <aside className="guide-brief" aria-label="Before you start">
            <p className="guide-brief-title">Before you start</p>
            <ul>
              <li><b>Windows 10/11 x64</b> for the packaged desktop build. The browser studio also runs on Linux and macOS.</li>
              <li><b>One dataset</b> in text, JSONL, CSV, Parquet or Arrow format.</li>
              <li><b>A GPU if you have one.</b> CPU training works, but keep the model and sequence length small.</li>
              <li><b>Disk space</b> for the dataset, token cache and checkpoints. Checkpoints are the large files.</li>
            </ul>
            <p className="guide-brief-foot">Everything stays on this machine unless you export it.</p>
          </aside>
        </header>

        <nav className="wrap guide-toc" aria-label="Guide contents">
          <span>Contents</span>
          <a href="#install">01 Install</a>
          <a href="#architecture">02 Architecture</a>
          <a href="#data">03 Data</a>
          <a href="#tokenize">04 Tokenize</a>
          <a href="#pretrain">05 Pretrain</a>
          <a href="#sft">06 SFT</a>
          <a href="#evaluate">07 Evaluate</a>
          <a href="#keep">08 Keep and resume</a>
        </nav>

        <div className="wrap guide-steps">
          <GuideStep id="install" number="01" label="Install" title="Install once, then stay local.">
            <p>The desktop app is the intended path. From a repository checkout, run the setup once, then launch the desktop shell:</p>
            <pre className="guide-code"><code>{"setup.bat\nrun_desktop.bat"}</code></pre>
            <p>If you already have the packaged build, launch <code>llm-maker.exe</code>. For the browser studio, run:</p>
            <pre className="guide-code"><code>{"run_studio.bat"}</code></pre>
            <p>Then open <a href="http://localhost:8080">http://localhost:8080</a>. On Linux or macOS, use <code>run_studio.sh</code> for the same browser studio. The desktop shell is the same interface, with the worker and database managed for you.</p>
            <div className="guide-note"><b>Unsigned Windows build.</b> SmartScreen may warn on a downloaded installer. Verify the published SHA-256 before you run it.</div>
            <ul className="guide-check">
              <li><b>The app opens.</b> The Overview should show the detected device and its memory.</li>
              <li><b>No setup errors.</b> If the launcher reports a failure, fix that before importing data.</li>
            </ul>
          </GuideStep>

          <GuideStep id="architecture" number="02" label="Architecture" title="Start from a model you can afford to train.">
            <p>Open <b>Network Studio</b> and start from the <b>RoCanon-48.5M Baseline</b> template, or load a network you already trust. The graph defines the model; the template is a starting point, not a rule.</p>
            <p>For a first run, change one thing at a time. More layers or a wider model costs VRAM and time; it does not fix a noisy dataset.</p>
            <p>Save the network before you leave the studio so it appears in the training form.</p>
            <div className="guide-note"><b>Small is a feature.</b> A small base model learns format and tone long before it learns reliable facts. That is what SFT and evaluation are for.</div>
            <ul className="guide-check">
              <li><b>The network is saved.</b> Training needs a stable architecture it can load again.</li>
              <li><b>The parameter count fits your device.</b> Leave headroom for activations and the optimizer.</li>
            </ul>
          </GuideStep>

          <GuideStep id="data" number="03" label="Data" title="Import data you would be willing to read.">
            <p>Open <b>Datasets</b>, then choose <b>Upload Training Dataset</b>. Leave <b>Clean on import</b> selected.</p>
            <p>The importer accepts <code>.txt</code>, <code>.jsonl</code>, <code>.csv</code>, <code>.parquet</code> and <code>.arrow</code>. It keeps the original under <code>raw/</code> and makes the cleaned copy the visible dataset.</p>
            <p>Cleaning repairs mojibake, removes standalone <code>nan</code> lines and missing or <code>NaN</code> fields, masks PII, filters quality and language, and drops exact and near duplicates.</p>
            <p>If a dataset was imported earlier, use its <b>Clean</b> action now. Do not train on the raw copy and hope the problem disappears.</p>
            <div className="guide-note"><b>A <code>nan</code> is a missing value, not a token.</b> In enough targets, the model learns to emit it. Cleaning is cheaper than unlearning it.</div>
            <ul className="guide-check">
              <li><b>The record count is plausible.</b> A huge drop means the filters were too strict or the format was misread.</li>
              <li><b>Open a few records.</b> Check the language, the fields and the line endings.</li>
              <li><b>No standalone <code>nan</code> remains.</b> Search the cleaned text before you tokenize.</li>
            </ul>
          </GuideStep>

          <GuideStep id="tokenize" number="04" label="Tokenize" title="Turn the text into one stable token cache.">
            <p>With the dataset selected, choose <b>Tokenize</b> and wait for the live operation to finish. The row changes to <b>Tokenized</b>.</p>
            <p>In <b>Training</b>, confirm the token count and the steps-per-epoch estimate. Both come from the tokenized train split and your sequence and batch settings.</p>
            <p>If the estimate is zero, the dataset is not tokenized or the sequence length is larger than the data. Fix that before starting a run.</p>
            <div className="guide-note"><b>Retokenize after cleaning.</b> A changed dataset needs a fresh cache; an old token file will not contain the repaired text.</div>
            <ul className="guide-check">
              <li><b>Status is Tokenized.</b> The token count is visible in the dataset row.</li>
              <li><b>The step estimate is not zero.</b> Check the sequence length and batch settings if it is.</li>
            </ul>
          </GuideStep>

          <GuideStep id="pretrain" number="05" label="Pretrain" title="Teach the base language before you teach it to answer.">
            <p>Open <b>Training</b>. Select the network and the tokenized dataset, then press <b>⚡ Pretrain Preset</b>.</p>
            <p>For a first run, keep <b>Epochs</b> at 2–3 and leave the preset learning rates and WSD split alone. Set a checkpoint interval you can afford to lose, and a validation interval that produces several points.</p>
            <p>Check that the planned step count is greater than zero, then press <b>Start Training</b>. The run writes checkpoints, metrics and a best validation model.</p>
            <p>Watch validation loss, the learning-rate phase and VRAM. A downward loss with no NaNs is the signal to continue; a flat or rising loss is the signal to stop and inspect the data.</p>
            <div className="guide-note"><b>The base is not a chatbot yet.</b> Pretraining teaches next-token prediction. It will continue a prompt as a document, not answer a question.</div>
            <ul className="guide-check">
              <li><b>Loss trends down.</b> It does not need to reach zero, and a sudden spike deserves a look at the batch.</li>
              <li><b>Checkpoints appear.</b> A run with no saved checkpoint is one crash away from starting over.</li>
              <li><b><code>best_model.pt</code> appears.</b> The best checkpoint is replaced only by a better validation result.</li>
            </ul>
          </GuideStep>

          <GuideStep id="sft" number="06" label="SFT" title="Fine-tune the base to hold a conversation.">
            <p>When the pretrain run completes, stay in <b>Training</b>. Choose the instruction dataset, press <b>⚡ SFT Preset</b>, then set <b>Resume From Checkpoint</b> to the newest pretrain checkpoint.</p>
            <p>Press <b>Start Training</b>. The run continues from those weights as a new phase, with its own warmup, stable and decay window and its own epoch budget.</p>
            <p>SFT masks the prompt and scores the assistant tokens. The model learns the reply format and the instruction-following behavior; it does not relearn the base language from scratch.</p>
            <div className="guide-note"><b>Use a checkpoint saved by this build.</b> Older checkpoints have no phase metadata, so the app falls back to the global counters. That is the case where SFT can open at the end of the learning-rate curve.</div>
            <ul className="guide-check">
              <li><b>The phase is new.</b> The run should show a fresh warmup rather than a decay that starts low.</li>
              <li><b>Answers stay in one turn.</b> The model should stop when the reply ends, not continue as a document.</li>
              <li><b>No <code>nan</code> or repeated filler.</b> If either appears, stop and inspect the instruction data.</li>
            </ul>
          </GuideStep>

          <GuideStep id="evaluate" number="07" label="Evaluate" title="Read the outputs before you call it finished.">
            <p>Open <b>Evaluation</b>. Choose the checkpoint, keep <b>Chat Template</b> on, and set the temperature low, around 0.2 to 0.5, for a first read.</p>
            <p>Ask three things: a question the data should cover, a question it cannot know, and a prompt with an obvious format. Then use <b>Compare</b> with a second checkpoint and run the benchmark suite.</p>
            <p>Open <b>Runs</b> to check the loss curve and the exact checkpoint that was selected. A checkpoint can look good on validation and still fail in a conversation.</p>
            <div className="guide-note"><b>Read outputs, not just loss.</b> Validation loss ranks checkpoints. It does not prove the model is useful, honest or safe.</div>
            <ul className="guide-check">
              <li><b>Format is stable.</b> One answer, one ending, no raw template markers.</li>
              <li><b>Unknown questions are handled honestly.</b> A confident wrong answer is a data or evaluation problem, not a feature.</li>
              <li><b>The better checkpoint is better on the same prompts.</b> Change one variable when you compare.</li>
            </ul>
          </GuideStep>

          <GuideStep id="keep" number="08" label="Keep and resume" title="Keep the useful files, and know how to continue.">
            <p>From <b>Runs</b>, press <b>Resume</b> on a checkpoint. The app restores the weights and optimizer and starts the original run again, so the chart continues instead of starting a second run.</p>
            <p>The same operation is available in <b>Training</b> under <b>Resume From Checkpoint</b>, followed by <b>Start Training</b>. Use that path when you want to change the dataset or duration first.</p>
            <div className="guide-note"><b>Do not resume over a live run.</b> If a run is running, paused or stopping, the automatic resume refuses to replace it. Stop the current run first.</div>
            <p>Checkpoints follow the retention window in <b>Settings</b>: the best validation model, the most recent N step checkpoints, and one anchor from each completed phase. The default is four recent checkpoints; the minimum is one.</p>
            <p>Export from <b>Models</b> as GGUF for llama.cpp when the architecture uses the Llama layout, or keep the RoCanon layout for lossless weights. Export the whole workspace from <b>Settings</b> when you need a portable copy.</p>
            <ul className="guide-check">
              <li><b>Resume continues the step count.</b> The original run and chart history should survive.</li>
              <li><b>An anchor exists after a phase change.</b> That is the pretrain base you can return to.</li>
              <li><b>The exported file opens.</b> Test the export before you delete the checkpoint it came from.</li>
            </ul>
          </GuideStep>
        </div>

        <section className="wrap guide-section">
          <div className="section-head">
            <h2 id="troubleshooting">When something looks wrong.</h2>
            <p>Most first-run failures are data, cache or resume-state problems. Check these before changing the architecture.</p>
          </div>
          <div className="guide-table-wrap">
            <table className="ledger guide-table">
              <caption>What to check first.</caption>
              <thead>
                <tr><th scope="col">Symptom</th><th scope="col">Likely cause</th><th scope="col">What to do</th></tr>
              </thead>
              <tbody>
                <tr><td>The model writes <code>nan</code></td><td>The dataset or an old token cache contained missing values.</td><td>Clean the dataset, retokenize, and retrain from a clean checkpoint. A model that already learned the pattern will not forget it in a short SFT.</td></tr>
                <tr><td>Resume does nothing</td><td>A run is already running, paused or stopping.</td><td>Stop the current run, then press Resume again. The button is guarded so it cannot replace a live worker.</td></tr>
                <tr><td>SFT opens at the end of the learning-rate curve</td><td>The checkpoint predates phase-relative schedules.</td><td>Start SFT from a checkpoint saved by the current build, or start a new phase. The schedule is relative to the phase now.</td></tr>
                <tr><td>The pretrain checkpoint is gone</td><td>Older retention rules pruned it when SFT began.</td><td>It cannot be recovered from the current workspace. Current builds keep one anchor per completed phase; retrain the base once and keep that checkpoint.</td></tr>
                <tr><td>CUDA runs out of memory</td><td>The batch, sequence length or model is too large.</td><td>Lower micro-batch size, raise gradient accumulation, shorten sequence length, or choose a smaller network. Close other GPU apps.</td></tr>
                <tr><td>A dataset operation seems stuck</td><td>Cleaning or tokenization is still running in the background.</td><td>Open the Datasets operation list. Progress survives a view change and a page reload; wait for the status before starting a run.</td></tr>
              </tbody>
            </table>
          </div>
        </section>

        <section className="wrap guide-section">
          <div className="section-head">
            <h2 id="glossary">The words the app uses.</h2>
            <p>Short definitions, so the training form and the checkpoint table mean the same thing.</p>
          </div>
          <dl className="guide-terms">
            <div><dt>Step</dt><dd>One optimizer update over one batch. The loss chart is drawn in steps.</dd></div>
            <div><dt>Epoch</dt><dd>One full pass over the training split. The app turns epochs into a step budget.</dd></div>
            <div><dt>Phase</dt><dd>A pretrain or SFT stage with its own schedule, token count and epoch budget.</dd></div>
            <div><dt>Checkpoint</dt><dd>A saved model and optimizer state that can continue the run.</dd></div>
            <div><dt>Best checkpoint</dt><dd>The single file with the lowest validation loss so far.</dd></div>
            <div><dt>Anchor</dt><dd>The newest checkpoint from a completed phase, kept outside the rolling window.</dd></div>
          </dl>
        </section>

        <section className="wrap guide-end">
          <p className="kicker">After the first run</p>
          <h2>Change one variable at a time.</h2>
          <p>Once you have a baseline that trains, evaluates and resumes, improve it in small steps: cleaner data, more steps, a larger model, a different schedule. Keep the run that worked so you can compare against it.</p>
          <div className="actions">
            <a className="button" href={`${basePath}/changelog/`}>Read the changelog</a>
            <a className="text-link" href={REPO}>View the repository</a>
          </div>
        </section>
      </main>
    </div>
  );
}

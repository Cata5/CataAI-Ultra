import { ThemeToggle } from "./ThemeToggle";

const basePath = process.env.NEXT_PUBLIC_BASE_PATH ?? "";

type Page = "home" | "guide" | "changelog";

export function SiteRail({ page }: { page: Page }) {
  const home = page === "home" ? "" : `${basePath}/`;

  return (
    <aside className="rail">
      <a className="brand" href={page === "home" ? "#top" : `${basePath}/`} aria-label="LLM Maker home">
        <span className="brand-mark" aria-hidden="true">LM</span>
        <span>LLM Maker</span>
      </a>

      <nav aria-label="Sections">
        <a href={`${home}#overview`}>Overview</a>
        <a href={`${home}#audit`}>Audit</a>
        <a href={`${home}#roadmap`}>Roadmap</a>
        <a href={`${home}#screens`}>Screens</a>
        <a href={`${basePath}/guide/`} aria-current={page === "guide" ? "page" : undefined}>
          Guide
        </a>
        <a href={`${basePath}/changelog/`} aria-current={page === "changelog" ? "page" : undefined}>
          Changelog
        </a>
      </nav>

      <div className="rail-foot">
        <ThemeToggle />
        <div className="live"><span className="dot" aria-hidden="true" />v0.1.0 · experimental</div>
        <div>Windows 10/11 · x64</div>
        <div>Local-first. No telemetry.</div>
      </div>
    </aside>
  );
}

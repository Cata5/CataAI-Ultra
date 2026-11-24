import type { Metadata } from "next";
import { ChangelogTabs } from "@/components/ChangelogTabs";
import { SiteRail } from "@/components/SiteRail";
import { experimentalChanges, releaseChanges } from "@/lib/changelog";

export const metadata: Metadata = {
  title: "Changelog — LLM Maker",
  description:
    "Release and experimental change history for LLM Maker, sourced from the public GitHub repository.",
};

export default function ChangelogPage() {
  return (
    <div className="shell">
      <SiteRail page="changelog" />

      <main>
        <section className="wrap changelog-hero">
          <p className="kicker">Change log</p>
          <h1>Every change, both channels.</h1>
          <p className="lede">
            Release history follows <code>main</code>. Experimental lists only the
            commits that are not on <code>main</code> yet, so the two tabs never repeat
            the same change.
          </p>
          <p className="changelog-note">
            {releaseChanges.length} release commits · {experimentalChanges.length} newer
            experimental commits · sourced from{" "}
            <a href="https://github.com/Cata5/CataAI-Ultra">GitHub</a>. No tagged releases
            exist yet.
          </p>
          <ChangelogTabs />
        </section>
      </main>
    </div>
  );
}

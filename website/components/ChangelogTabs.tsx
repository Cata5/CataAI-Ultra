"use client";

import { useState, type KeyboardEvent } from "react";
import {
  experimentalChanges,
  formatChangeDate,
  releaseChanges,
  REPO,
  type ChangeEntry,
} from "@/lib/changelog";

type Channel = "release" | "experimental";

const channels: Record<Channel, { label: string; note: string; entries: ChangeEntry[] }> = {
  release: {
    label: "Release",
    note: "The main branch. These are the changes in the published history.",
    entries: releaseChanges,
  },
  experimental: {
    label: "Experimental",
    note: "Commits on experimental that are not on main yet. Release changes are not repeated here.",
    entries: experimentalChanges,
  },
};

function ChangeRow({ entry }: { entry: ChangeEntry }) {
  const href = `${REPO}/commit/${entry.sha}`;
  return (
    <li className="change">
      <div className="change-meta">
        <time dateTime={entry.date}>{formatChangeDate(entry.date)}</time>
        <span className={`change-kind kind-${entry.kind.toLowerCase()}`}>{entry.kind}</span>
      </div>
      <div className="change-copy">
        <h3>{entry.title}</h3>
        <p>{entry.summary}</p>
        <a className="commit-link" href={href} target="_blank" rel="noreferrer">
          <code>{entry.sha}</code> on GitHub <span aria-hidden="true">↗</span>
        </a>
      </div>
    </li>
  );
}

const channelOrder = Object.keys(channels) as Channel[];

export function ChangelogTabs() {
  const [channel, setChannel] = useState<Channel>("release");
  const active = channels[channel];

  const onTabKeyDown = (event: KeyboardEvent<HTMLButtonElement>) => {
    const index = channelOrder.indexOf(channel);
    let next: Channel | null = null;

    if (event.key === "ArrowRight") next = channelOrder[(index + 1) % channelOrder.length];
    if (event.key === "ArrowLeft") next = channelOrder[(index - 1 + channelOrder.length) % channelOrder.length];
    if (event.key === "Home") next = channelOrder[0];
    if (event.key === "End") next = channelOrder[channelOrder.length - 1];
    if (!next) return;

    event.preventDefault();
    setChannel(next);
    document.getElementById(`changelog-tab-${next}`)?.focus();
  };

  return (
    <div className="changelog-tabs">
      <div className="tablist" role="tablist" aria-label="Changelog channel">
        {channelOrder.map((key) => (
          <button
            key={key}
            type="button"
            role="tab"
            id={`changelog-tab-${key}`}
            aria-controls="changelog-panel"
            aria-selected={channel === key}
            tabIndex={channel === key ? 0 : -1}
            className="tab"
            onClick={() => setChannel(key)}
            onKeyDown={onTabKeyDown}
          >
            {channels[key].label}
            <span className="tab-count">{channels[key].entries.length}</span>
          </button>
        ))}
      </div>

      <div
        role="tabpanel"
        id="changelog-panel"
        aria-labelledby={`changelog-tab-${channel}`}
        className="changelog-panel"
      >
        <p className="channel-note">{active.note}</p>
        <ol className="timeline">
          {active.entries.map((entry) => (
            <ChangeRow key={`${channel}-${entry.sha}`} entry={entry} />
          ))}
        </ol>
      </div>
    </div>
  );
}

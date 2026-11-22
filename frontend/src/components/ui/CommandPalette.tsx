"use client";

import React, { useEffect, useMemo, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import {
  Search,
  LayoutDashboard,
  Network,
  Play,
  Database,
  FlaskConical,
  History,
  Settings,
  CornerDownLeft,
} from "lucide-react";

interface CommandPaletteProps {
  isOpen: boolean;
  onClose: () => void;
}

export const CommandPalette: React.FC<CommandPaletteProps> = ({ isOpen, onClose }) => {
  const router = useRouter();
  const [query, setQuery] = useState("");
  const [activeIndex, setActiveIndex] = useState(0);
  const listRef = useRef<HTMLDivElement>(null);

  const commands = useMemo(
    () => [
      { label: "Overview / Command Center", path: "/", icon: <LayoutDashboard className="w-4 h-4 text-[#45CFFF]" /> },
      { label: "Network Studio (Architecture Designer)", path: "/studio", icon: <Network className="w-4 h-4 text-[#9B8CFF]" /> },
      { label: "Training & Experiment Control", path: "/training", icon: <Play className="w-4 h-4 text-[#35D49A]" /> },
      { label: "Datasets & Data Explorer", path: "/datasets", icon: <Database className="w-4 h-4 text-[#F2B85B]" /> },
      { label: "Evaluation Lab & Inference", path: "/evaluation", icon: <FlaskConical className="w-4 h-4 text-[#80DEFF]" /> },
      { label: "Runs, Checkpoints & Analytics", path: "/runs", icon: <History className="w-4 h-4 text-[#9B8CFF]" /> },
      { label: "Settings & GPU Monitoring", path: "/settings", icon: <Settings className="w-4 h-4 text-[#91A0B4]" /> },
    ],
    []
  );

  const filtered = useMemo(
    () => commands.filter((c) => c.label.toLowerCase().includes(query.toLowerCase())),
    [commands, query]
  );

  const active = Math.min(activeIndex, Math.max(0, filtered.length - 1));

  const run = (path: string) => {
    router.push(path);
    onClose();
  };

  useEffect(() => {
    if (!isOpen) return;
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape" || ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k")) {
        e.preventDefault();
        onClose();
        return;
      }
      if (e.key === "ArrowDown") {
        e.preventDefault();
        setActiveIndex((i) => (filtered.length ? (i + 1) % filtered.length : 0));
      } else if (e.key === "ArrowUp") {
        e.preventDefault();
        setActiveIndex((i) => (filtered.length ? (i - 1 + filtered.length) % filtered.length : 0));
      } else if (e.key === "Enter" && filtered[active]) {
        e.preventDefault();
        run(filtered[active].path);
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  });

  useEffect(() => {
    const node = listRef.current?.querySelector<HTMLElement>(`[data-index="${active}"]`);
    node?.scrollIntoView({ block: "nearest" });
  }, [active]);

  if (!isOpen) return null;

  return (
    <div
      className="fixed inset-0 z-50 flex items-start justify-center pt-24 p-4 bg-[rgba(5,8,13,0.8)] backdrop-blur-sm"
      onMouseDown={(e) => {
        if (e.target === e.currentTarget) onClose();
      }}
    >
      <div
        role="dialog"
        aria-modal="true"
        aria-label="Command palette"
        className="w-full max-w-lg overflow-hidden rounded-[12px] border border-[var(--border-strong)] bg-[var(--bg-surface)] shadow-[0_24px_80px_-20px_rgba(0,0,0,0.95)] animate-in fade-in zoom-in-95 duration-150"
      >
        <div className="flex items-center gap-2.5 p-3.5 border-b border-[var(--border-subtle)]">
          <Search className="w-4 h-4 text-[#45CFFF]" aria-hidden="true" />
          <label htmlFor="command-search" className="sr-only">
            Search commands
          </label>
          <input
            id="command-search"
            autoFocus
            type="text"
            value={query}
            onChange={(e) => {
              setQuery(e.target.value);
              setActiveIndex(0);
            }}
            placeholder="Type a command or jump to workspace…"
            className="w-full bg-transparent text-[13px] font-mono text-[#ECF3FC] placeholder-[#5E7188] outline-none"
            role="combobox"
            aria-expanded="true"
            aria-autocomplete="list"
            aria-controls="command-list"
            aria-activedescendant={filtered[active] ? `command-${active}` : undefined}
          />
          <kbd className="kbd">Esc</kbd>
        </div>

        <div
          id="command-list"
          ref={listRef}
          role="listbox"
          aria-label="Commands"
          className="max-h-72 overflow-y-auto p-2 space-y-0.5"
        >
          {filtered.length === 0 ? (
            <div className="p-6 text-center space-y-1">
              <p className="font-mono text-[11px] font-semibold uppercase tracking-wider text-[#91A0B4]">
                No matching commands
              </p>
              <p className="text-[11px] text-[#5E7188]">Try “training”, “datasets” or “settings”.</p>
            </div>
          ) : (
            filtered.map((item, idx) => (
              <button
                key={item.path}
                id={`command-${idx}`}
                data-index={idx}
                role="option"
                aria-selected={idx === active}
                onMouseEnter={() => setActiveIndex(idx)}
                onClick={() => run(item.path)}
                className={`w-full flex items-center gap-3 px-3 py-2 rounded-md text-left text-[12px] font-mono transition-colors ${
                  idx === active
                    ? "bg-[var(--bg-elevated)] text-[#ECF3FC]"
                    : "text-[#91A0B4] hover:bg-[var(--bg-elevated)]/60 hover:text-[#ECF3FC]"
                }`}
              >
                <span aria-hidden="true">{item.icon}</span>
                <span className="flex-1 truncate">{item.label}</span>
                {idx === active && (
                  <CornerDownLeft className="w-3.5 h-3.5 text-[#5E7188]" aria-hidden="true" />
                )}
              </button>
            ))
          )}
        </div>

        <div className="flex items-center gap-3 px-3.5 py-2 border-t border-[var(--border-subtle)] text-[10px] font-mono text-[#5E7188]">
          <span className="flex items-center gap-1">
            <kbd className="kbd">↑</kbd>
            <kbd className="kbd">↓</kbd> navigate
          </span>
          <span className="flex items-center gap-1">
            <kbd className="kbd">↵</kbd> open
          </span>
        </div>
      </div>
    </div>
  );
};

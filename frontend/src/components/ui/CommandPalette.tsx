"use client";

import React, { useEffect, useMemo, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import {
  Search,
  LayoutDashboard,
  FolderKanban,
  Boxes,
  Network,
  Play,
  Database,
  FlaskConical,
  History,
  Bookmark,
  BarChart2,
  Cpu,
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
      { label: "Overview", path: "/", icon: <LayoutDashboard className="w-4 h-4 text-[#A3A3A3]" /> },
      { label: "Projects", path: "/projects", icon: <FolderKanban className="w-4 h-4 text-[#A3A3A3]" /> },
      { label: "Models", path: "/models", icon: <Boxes className="w-4 h-4 text-[#A3A3A3]" /> },
      { label: "Network Studio", path: "/studio", icon: <Network className="w-4 h-4 text-[#A3A3A3]" /> },
      { label: "Datasets", path: "/datasets", icon: <Database className="w-4 h-4 text-[#A3A3A3]" /> },
      { label: "Training Console", path: "/training", icon: <Play className="w-4 h-4 text-[#A3A3A3]" /> },
      { label: "Evaluation Lab", path: "/evaluation", icon: <FlaskConical className="w-4 h-4 text-[#A3A3A3]" /> },
      { label: "Experiments & Runs", path: "/runs", icon: <History className="w-4 h-4 text-[#A3A3A3]" /> },
      { label: "Checkpoints", path: "/checkpoints", icon: <Bookmark className="w-4 h-4 text-[#A3A3A3]" /> },
      { label: "Analytics", path: "/analytics", icon: <BarChart2 className="w-4 h-4 text-[#A3A3A3]" /> },
      { label: "Hardware Telemetry", path: "/settings?tab=hardware", icon: <Cpu className="w-4 h-4 text-[#A3A3A3]" /> },
      { label: "Settings", path: "/settings", icon: <Settings className="w-4 h-4 text-[#A3A3A3]" /> },
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
      className="fixed inset-0 z-50 flex items-start justify-center pt-20 p-4 bg-black/60 backdrop-blur-sm"
      onMouseDown={(e) => {
        if (e.target === e.currentTarget) onClose();
      }}
    >
      <div
        role="dialog"
        aria-modal="true"
        aria-label="Command palette"
        className="w-full max-w-md overflow-hidden rounded-lg border border-[#333333] bg-[#202020] shadow-xl"
      >
        <div className="flex items-center gap-2 p-3 border-b border-[#262626] bg-[#1B1B1B]">
          <Search className="w-3.5 h-3.5 text-[#737373]" aria-hidden="true" />
          <label htmlFor="command-search" className="sr-only">
            Search commands
          </label>
          <input
            id="command-search"
            type="text"
            value={query}
            onChange={(e) => {
              setQuery(e.target.value);
              setActiveIndex(0);
            }}
            placeholder="Type a command or jump to page…"
            autoFocus
            className="w-full bg-transparent text-xs text-[#F2F2F2] placeholder-[#737373] outline-none font-mono"
          />
          <kbd className="kbd text-[10px]">ESC</kbd>
        </div>

        <div ref={listRef} className="max-h-64 overflow-y-auto p-1.5 space-y-0.5">
          {filtered.length === 0 ? (
            <div className="p-4 text-center text-xs text-[#737373] font-mono">
              No matching commands.
            </div>
          ) : (
            filtered.map((item, idx) => (
              <button
                key={item.path + item.label}
                data-index={idx}
                type="button"
                onClick={() => run(item.path)}
                className={`w-full flex items-center justify-between px-2.5 py-1.5 rounded text-xs transition-colors font-mono ${
                  idx === active
                    ? "bg-[#262626] text-[#F2F2F2]"
                    : "text-[#A3A3A3] hover:bg-[#262626] hover:text-[#F2F2F2]"
                }`}
              >
                <span className="flex items-center gap-2 truncate">
                  {item.icon}
                  <span className="truncate">{item.label}</span>
                </span>
                {idx === active && (
                  <CornerDownLeft className="w-3 h-3 text-[#737373]" aria-hidden="true" />
                )}
              </button>
            ))
          )}
        </div>
      </div>
    </div>
  );
};

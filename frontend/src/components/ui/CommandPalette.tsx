"use client";

import React, { useState, useEffect } from "react";
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
  X,
} from "lucide-react";

interface CommandPaletteProps {
  isOpen: boolean;
  onClose: () => void;
}

export const CommandPalette: React.FC<CommandPaletteProps> = ({ isOpen, onClose }) => {
  const router = useRouter();
  const [query, setQuery] = useState("");

  const commands = [
    { label: "Overview / Command Center", path: "/", icon: <LayoutDashboard className="w-4 h-4 text-cyan-400" /> },
    { label: "Network Studio (Architecture Designer)", path: "/studio", icon: <Network className="w-4 h-4 text-indigo-400" /> },
    { label: "Training & Experiment Control", path: "/training", icon: <Play className="w-4 h-4 text-emerald-400" /> },
    { label: "Datasets & Data Explorer", path: "/datasets", icon: <Database className="w-4 h-4 text-amber-400" /> },
    { label: "Evaluation Lab & Inference", path: "/evaluation", icon: <FlaskConical className="w-4 h-4 text-sky-400" /> },
    { label: "Runs, Checkpoints & Analytics", path: "/runs", icon: <History className="w-4 h-4 text-purple-400" /> },
    { label: "Settings & GPU Monitoring", path: "/settings", icon: <Settings className="w-4 h-4 text-slate-400" /> },
  ];

  const filtered = commands.filter((c) =>
    c.label.toLowerCase().includes(query.toLowerCase())
  );

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key === "k") {
        e.preventDefault();
        onClose();
      }
      if (e.key === "Escape") {
        onClose();
      }
    };
    if (isOpen) {
      window.addEventListener("keydown", handleKeyDown);
    }
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [isOpen, onClose]);

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-start justify-center pt-24 p-4 bg-black/70 backdrop-blur-sm">
      <div className="glass-panel border border-cyan-500/30 rounded-xl max-w-lg w-full shadow-2xl overflow-hidden animate-in fade-in zoom-in-95 duration-150">
        <div className="flex items-center gap-2.5 p-3.5 border-b border-slate-800">
          <Search className="w-4 h-4 text-cyan-400" />
          <input
            autoFocus
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Type a command or jump to workspace..."
            className="w-full bg-transparent text-sm font-mono text-slate-100 placeholder-slate-400 focus:outline-none"
          />
          <button
            onClick={onClose}
            className="p-1 rounded text-slate-400 hover:text-slate-200"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        <div className="max-h-72 overflow-y-auto p-2 space-y-1">
          {filtered.length === 0 ? (
            <div className="p-4 text-center font-mono text-xs text-slate-400">
              No matching commands.
            </div>
          ) : (
            filtered.map((item, idx) => (
              <button
                key={idx}
                onClick={() => {
                  router.push(item.path);
                  onClose();
                }}
                className="w-full flex items-center gap-3 px-3 py-2 rounded-lg text-left text-xs font-mono text-slate-300 hover:bg-slate-800/80 hover:text-cyan-300 transition-colors"
              >
                {item.icon}
                <span className="flex-1 font-medium">{item.label}</span>
                <span className="text-[10px] text-slate-400 font-mono">Jump</span>
              </button>
            ))
          )}
        </div>

        <div className="p-2 border-t border-slate-800/80 bg-slate-900/60 flex items-center justify-between text-[10px] font-mono text-slate-400 px-3">
          <span>Navigate with mouse or arrow keys</span>
          <span>ESC to close</span>
        </div>
      </div>
    </div>
  );
};

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
    { label: "Overview / Command Center", path: "/", icon: <LayoutDashboard className="w-4 h-4 text-[#45CFFF]" /> },
    { label: "Network Studio (Architecture Designer)", path: "/studio", icon: <Network className="w-4 h-4 text-[#9B8CFF]" /> },
    { label: "Training & Experiment Control", path: "/training", icon: <Play className="w-4 h-4 text-[#35D49A]" /> },
    { label: "Datasets & Data Explorer", path: "/datasets", icon: <Database className="w-4 h-4 text-[#F2B85B]" /> },
    { label: "Evaluation Lab & Inference", path: "/evaluation", icon: <FlaskConical className="w-4 h-4 text-[#80DEFF]" /> },
    { label: "Runs, Checkpoints & Analytics", path: "/runs", icon: <History className="w-4 h-4 text-[#9B8CFF]" /> },
    { label: "Settings & GPU Monitoring", path: "/settings", icon: <Settings className="w-4 h-4 text-[#91A0B4]" /> },
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
    <div className="fixed inset-0 z-50 flex items-start justify-center pt-24 p-4 bg-black/80 backdrop-blur-md">
      <div className="bg-[#0E1420] border border-[#384C66] rounded-xl max-w-lg w-full shadow-2xl overflow-hidden animate-in fade-in zoom-in-95 duration-150">
        <div className="flex items-center gap-2.5 p-3.5 border-b border-[#273548]">
          <Search className="w-4 h-4 text-[#45CFFF]" />
          <input
            autoFocus
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Type a command or jump to workspace..."
            className="w-full bg-transparent text-sm font-mono text-[#ECF3FC] placeholder-[#5E7188] focus:outline-none"
          />
          <button
            onClick={onClose}
            className="p-1 rounded text-[#91A0B4] hover:text-[#ECF3FC]"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        <div className="max-h-72 overflow-y-auto p-2 space-y-1">
          {filtered.length === 0 ? (
            <div className="p-4 text-center font-mono text-xs text-[#5E7188]">
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
                className="w-full flex items-center gap-3 px-3 py-2 rounded-lg text-left text-xs font-mono text-[#91A0B4] hover:bg-[#182333] hover:text-[#45CFFF] transition-colors"
              >
                {item.icon}
                <span className="flex-1 font-medium">{item.label}</span>
                <span className="text-[10px] text-[#5E7188] font-mono">Jump</span>
              </button>
            ))
          )}
        </div>
      </div>
    </div>
  );
};

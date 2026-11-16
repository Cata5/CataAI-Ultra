"use client";

import React, { Component, ReactNode } from "react";
import dynamic from "next/dynamic";
import { LoadingState } from "@/components/ui/LoadingState";

interface Props {
  children: ReactNode;
}

interface State {
  hasError: boolean;
  error: Error | null;
}

class StudioErrorBoundary extends Component<Props, State> {
  constructor(props: Props) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, errorInfo: any) {
    console.error("Studio render exception:", error, errorInfo);
  }

  render() {
    if (this.state.hasError) {
      return (
        <div className="h-[60vh] flex flex-col items-center justify-center space-y-4 font-mono text-center p-6">
          <div className="p-3 rounded-full bg-rose-950/60 border border-rose-500/40 text-rose-400 text-lg">
            ⚠️
          </div>
          <h2 className="text-sm font-bold text-slate-100 uppercase">
            Architecture Canvas Initialization Warning
          </h2>
          <p className="text-xs text-slate-400 max-w-md">
            {this.state.error?.message || "An unexpected error occurred while rendering the graph."}
          </p>
          <button
            onClick={() => this.setState({ hasError: false, error: null })}
            className="px-4 py-2 rounded bg-cyan-600 hover:bg-cyan-500 text-white text-xs font-bold transition-all shadow-[0_0_10px_rgba(0,240,255,0.3)]"
          >
            RELOAD CANVAS
          </button>
        </div>
      );
    }
    return this.props.children;
  }
}

const NetworkStudioView = dynamic(
  () =>
    import("@/features/network-studio/NetworkStudioView").then(
      (mod) => mod.NetworkStudioView
    ),
  {
    ssr: false,
    loading: () => <LoadingState message="Initializing Architecture Designer..." />,
  }
);

export default function StudioPage() {
  return (
    <StudioErrorBoundary>
      <NetworkStudioView />
    </StudioErrorBoundary>
  );
}

"use client";

import React from "react";
import { Sidebar } from "./Sidebar";
import { Topbar } from "./Topbar";
import { StatusBar } from "./StatusBar";
import { I18nProvider } from "@/i18n";
import { DatasetOperationsProvider } from "@/features/datasets/DatasetOperations";

interface AppShellProps {
  children: React.ReactNode;
}

export const AppShell: React.FC<AppShellProps> = ({ children }) => {
  return (
    <I18nProvider>
      <DatasetOperationsProvider>
        <div className="flex h-screen w-screen overflow-hidden bg-[var(--bg-base)] text-[var(--text-primary)]">
          <Sidebar />
          <div className="flex flex-col flex-1 min-w-0 overflow-hidden">
            <Topbar />
            <main className="flex-1 overflow-y-auto p-4 md:p-5 relative bg-[var(--bg-base)]">
              {children}
            </main>
            <StatusBar />
          </div>
        </div>
      </DatasetOperationsProvider>
    </I18nProvider>
  );
};

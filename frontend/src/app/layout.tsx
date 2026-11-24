import type { Metadata } from "next";
import "./globals.css";
import { QueryProvider } from "@/components/providers/QueryProvider";
import { AppShell } from "@/components/layout/AppShell";
import { RuntimeSetupGate } from "@/components/onboarding/RuntimeSetupGate";

export const metadata: Metadata = {
  title: "LLM Maker — Neural Research Workstation",
  description:
    "Advanced Local Neural Network & LLM Training Platform, Architecture Studio & Telemetry Console",
  icons: {
    icon: "/logo.png",
    shortcut: "/logo.png",
    apple: "/logo.png",
  },
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="dark h-full antialiased">
      <body className="h-full bg-[var(--bg-base)] text-[var(--text-primary)] overflow-hidden">
        <QueryProvider>
          <RuntimeSetupGate>
            <AppShell>{children}</AppShell>
          </RuntimeSetupGate>
        </QueryProvider>
      </body>
    </html>
  );
}

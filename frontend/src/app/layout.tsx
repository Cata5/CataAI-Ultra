import type { Metadata } from "next";
import "./globals.css";
import { QueryProvider } from "@/components/providers/QueryProvider";
import { AppShell } from "@/components/layout/AppShell";

export const metadata: Metadata = {
  title: "LLM Maker — Neural Research Workstation",
  description:
    "Advanced Local Neural Network & LLM Training Platform, Architecture Studio & Telemetry Console",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="dark h-full antialiased">
      <body className="h-full bg-[#090D14] text-[#ECF3FC] overflow-hidden">
        <QueryProvider>
          <AppShell>{children}</AppShell>
        </QueryProvider>
      </body>
    </html>
  );
}

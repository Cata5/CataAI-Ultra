import React from "react";
import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import { StatusIndicator } from "@/components/ui/StatusIndicator";

describe("StatusIndicator", () => {
  it("exposes the run state as a status region with the uppercased label", () => {
    render(<StatusIndicator state="running" />);
    expect(screen.getByRole("status", { name: "Run state: RUNNING" })).toBeInTheDocument();
    expect(screen.getByText("RUNNING")).toBeInTheDocument();
  });

  it("falls back to the idle style for unknown states", () => {
    render(<StatusIndicator state="mystery" />);
    expect(screen.getByRole("status", { name: "Run state: MYSTERY" })).toBeInTheDocument();
  });
});

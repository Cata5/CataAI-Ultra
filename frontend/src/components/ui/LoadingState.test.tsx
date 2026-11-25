import React from "react";
import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import { LoadingState } from "@/components/ui/LoadingState";

describe("LoadingState", () => {
  it("renders the status region with the default message", () => {
    render(<LoadingState />);
    expect(screen.getByRole("status")).toBeInTheDocument();
    expect(screen.getByText(/Loading telemetry/)).toBeInTheDocument();
  });

  it("renders a custom message and hint", () => {
    render(<LoadingState message="Scanning checkpoints" hint="Usually fast" />);
    expect(screen.getByText("Scanning checkpoints")).toBeInTheDocument();
    expect(screen.getByText("Usually fast")).toBeInTheDocument();
  });
});

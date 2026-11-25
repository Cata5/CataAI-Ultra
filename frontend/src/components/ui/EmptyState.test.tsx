import React from "react";
import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import { EmptyState } from "@/components/ui/EmptyState";

describe("EmptyState", () => {
  it("renders the title and description", () => {
    render(<EmptyState title="No runs yet" description="Start a run to see metrics here." />);
    expect(screen.getByText("No runs yet")).toBeInTheDocument();
    expect(screen.getByText("Start a run to see metrics here.")).toBeInTheDocument();
  });

  it("renders the compact variant and an action slot", () => {
    render(
      <EmptyState title="Empty" compact action={<button type="button">Do the thing</button>} />
    );
    expect(screen.getByRole("button", { name: "Do the thing" })).toBeInTheDocument();
  });
});

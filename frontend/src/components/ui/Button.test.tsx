import React from "react";
import { describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen } from "@testing-library/react";
import { Button, buttonClasses } from "@/components/ui/Button";

describe("Button", () => {
  it("renders the label", () => {
    render(<Button>Save</Button>);
    expect(screen.getByRole("button", { name: "Save" })).toBeInTheDocument();
  });

  it("calls onClick once per click", () => {
    const onClick = vi.fn();
    render(<Button onClick={onClick}>Run</Button>);
    fireEvent.click(screen.getByRole("button", { name: "Run" }));
    expect(onClick).toHaveBeenCalledTimes(1);
  });

  it("is disabled and busy while loading", () => {
    render(<Button loading>Wait</Button>);
    const button = screen.getByRole("button", { name: "Wait" }) as HTMLButtonElement;
    expect(button).toBeDisabled();
    expect(button).toHaveAttribute("aria-busy", "true");
  });

  it("applies the variant classes", () => {
    expect(buttonClasses("primary")).toContain("btn-primary");
    expect(buttonClasses()).toContain("btn-secondary");
    expect(buttonClasses("ghost", "sm")).toContain("btn-sm");
  });
});

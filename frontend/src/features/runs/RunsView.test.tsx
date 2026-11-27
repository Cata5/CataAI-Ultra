import React from "react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { RunsView } from "@/features/runs/RunsView";
import { api } from "@/services/api";
import type { CheckpointItem } from "@/types/api";

const checkpoints: CheckpointItem[] = [
  {
    id: "cp-1",
    filename: "first.pt",
    step: 10,
    size_mb: 120,
    path: "D:\\checkpoints\\first.pt",
  },
  {
    id: "cp-2",
    filename: "second.pt",
    step: 20,
    size_mb: 240,
    path: "D:\\checkpoints\\second.pt",
  },
];

afterEach(() => {
  vi.restoreAllMocks();
});

describe("RunsView checkpoint bulk actions", () => {
  it("selects all checkpoints and deletes them together", async () => {
    vi.spyOn(api, "listRuns").mockResolvedValue([]);
    vi.spyOn(api, "listCheckpoints").mockResolvedValue(checkpoints);
    const deleteCheckpoints = vi
      .spyOn(api, "deleteCheckpoints")
      .mockResolvedValue({ deleted: ["cp-1", "cp-2"], failed: [] });

    render(<RunsView initialTab="checkpoints" />);

    expect(await screen.findByText("first.pt")).toBeInTheDocument();
    expect(screen.getAllByTitle("Resume Training from this checkpoint")[0]).toHaveAttribute(
      "href",
      "/training?resume_checkpoint=" + encodeURIComponent(checkpoints[0].path as string) + "&auto_resume=1"
    );
    fireEvent.click(screen.getByLabelText("Select checkpoint first.pt"));
    expect(screen.getByText("1 selected")).toBeInTheDocument();
    fireEvent.click(screen.getByLabelText("Select all checkpoints"));
    expect(screen.getByText("2 selected")).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "Delete selected" }));
    fireEvent.click(
      await screen.findByRole("button", { name: "Delete 2 Checkpoints" })
    );

    await waitFor(() => {
      expect(deleteCheckpoints).toHaveBeenCalledWith(["cp-1", "cp-2"]);
    });
    await waitFor(() => {
      expect(screen.queryByText("first.pt")).not.toBeInTheDocument();
      expect(screen.queryByText("second.pt")).not.toBeInTheDocument();
    });
  });
});

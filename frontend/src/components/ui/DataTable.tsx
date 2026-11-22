"use client";

import React, { useMemo, useState } from "react";
import { cn } from "@/lib/utils";
import { ChevronDown, ChevronUp, ChevronsUpDown, Search, X, Inbox } from "lucide-react";
import { Button } from "./Button";

export interface Column<T> {
  header: string;
  accessorKey?: keyof T;
  cell?: (item: T) => React.ReactNode;
  sortable?: boolean;
  className?: string;
}

interface DataTableProps<T> {
  data: T[];
  columns: Column<T>[];
  keyExtractor: (item: T) => string | number;
  searchPlaceholder?: string;
  searchFilter?: (item: T, query: string) => boolean;
  showSearch?: boolean;
  className?: string;
  emptyText?: string;
  pageSize?: number;
  /** Shown under the empty title, e.g. how to add the first record. */
  emptyHint?: string;
}

export function DataTable<T>({
  data,
  columns,
  keyExtractor,
  searchPlaceholder = "Search records…",
  searchFilter,
  showSearch = true,
  className,
  emptyText = "No records yet",
  pageSize = 15,
  emptyHint,
}: DataTableProps<T>) {
  const [search, setSearch] = useState("");
  const [sortKey, setSortKey] = useState<keyof T | null>(null);
  const [sortAsc, setSortAsc] = useState(false);
  const [currentPage, setCurrentPage] = useState(1);

  const filtered = useMemo(
    () =>
      data.filter((item) => {
        if (!search.trim()) return true;
        if (searchFilter) return searchFilter(item, search.toLowerCase());
        return JSON.stringify(item).toLowerCase().includes(search.toLowerCase());
      }),
    [data, search, searchFilter]
  );

  const sorted = useMemo(() => {
    if (!sortKey) return filtered;
    return [...filtered].sort((a, b) => {
      const aVal = a[sortKey];
      const bVal = b[sortKey];
      if (aVal === bVal) return 0;
      if (aVal === null || aVal === undefined) return 1;
      if (bVal === null || bVal === undefined) return -1;
      if (aVal < bVal) return sortAsc ? -1 : 1;
      return sortAsc ? 1 : -1;
    });
  }, [filtered, sortKey, sortAsc]);

  const totalPages = Math.max(1, Math.ceil(sorted.length / pageSize));
  const page = Math.min(currentPage, totalPages);
  const paginated = sorted.slice((page - 1) * pageSize, page * pageSize);

  const handleSort = (col: Column<T>) => {
    if (!col.sortable || !col.accessorKey) return;
    if (sortKey === col.accessorKey) {
      setSortAsc(!sortAsc);
    } else {
      setSortKey(col.accessorKey);
      setSortAsc(false);
    }
  };

  const description = (query: string) => {
    if (data.length === 0) {
      return { title: emptyText, hint: emptyHint ?? "Records will appear here once available." };
    }
    return {
      title: "No matching records",
      hint: `Nothing matches “${query}”. Try a different search term.`,
    };
  };

  const empty = description(search);

  return (
    <div className={cn("panel overflow-hidden flex flex-col", className)}>
      {showSearch && (
        <div className="p-3 border-b border-[var(--border-subtle)] bg-[var(--bg-subtle)]/60">
          <div className="relative">
            <Search
              className="pointer-events-none absolute left-2.5 top-1/2 -translate-y-1/2 w-3.5 h-3.5 text-[#5E7188]"
              aria-hidden="true"
            />
            <label htmlFor="data-table-search" className="sr-only">
              {searchPlaceholder}
            </label>
            <input
              id="data-table-search"
              type="search"
              value={search}
              onChange={(e) => {
                setSearch(e.target.value);
                setCurrentPage(1);
              }}
              placeholder={searchPlaceholder}
              className="input pl-8 pr-8 [&::-webkit-search-cancel-button]:hidden"
            />
            {search && (
              <button
                type="button"
                onClick={() => {
                  setSearch("");
                  setCurrentPage(1);
                }}
                className="icon-btn absolute right-0.5 top-1/2 -translate-y-1/2 h-7 w-7"
                aria-label="Clear search"
              >
                <X className="w-3.5 h-3.5" />
              </button>
            )}
          </div>
        </div>
      )}

      <div className="overflow-x-auto w-full">
        <table className="w-full text-left text-[12px] font-mono">
          <thead className="bg-[var(--bg-subtle)] text-[#91A0B4] border-b border-[var(--border-subtle)] uppercase tracking-wider text-[10px]">
            <tr>
              {columns.map((col, idx) => {
                const isSorted = col.sortable && sortKey === col.accessorKey;
                return (
                  <th
                    key={idx}
                    scope="col"
                    aria-sort={isSorted ? (sortAsc ? "ascending" : "descending") : undefined}
                    className={cn("px-3.5 py-2.5 font-bold whitespace-nowrap", col.className)}
                  >
                    {col.sortable ? (
                      <button
                        type="button"
                        onClick={() => handleSort(col)}
                        className={cn(
                          "inline-flex items-center gap-1 uppercase tracking-wider transition-colors hover:text-[#ECF3FC]",
                          isSorted && "text-[#45CFFF]"
                        )}
                      >
                        <span>{col.header}</span>
                        {isSorted ? (
                          sortAsc ? (
                            <ChevronUp className="w-3 h-3" aria-hidden="true" />
                          ) : (
                            <ChevronDown className="w-3 h-3" aria-hidden="true" />
                          )
                        ) : (
                          <ChevronsUpDown className="w-3 h-3 opacity-40" aria-hidden="true" />
                        )}
                      </button>
                    ) : (
                      <span>{col.header}</span>
                    )}
                  </th>
                );
              })}
            </tr>
          </thead>
          <tbody className="divide-y divide-[var(--border-subtle)] text-[#ECF3FC]">
            {paginated.length === 0 ? (
              <tr>
                <td colSpan={columns.length} className="px-4 py-12">
                  <div className="flex flex-col items-center justify-center gap-2 text-center">
                    {data.length === 0 ? (
                      <Inbox className="w-5 h-5 text-[#384C66]" aria-hidden="true" />
                    ) : (
                      <Search className="w-5 h-5 text-[#384C66]" aria-hidden="true" />
                    )}
                    <p className="text-[11px] font-semibold uppercase tracking-wider text-[#91A0B4]">
                      {empty.title}
                    </p>
                    <p className="max-w-sm text-[11px] text-[#5E7188]">{empty.hint}</p>
                  </div>
                </td>
              </tr>
            ) : (
              paginated.map((item) => (
                <tr key={keyExtractor(item)} className="hover:bg-[var(--bg-elevated)]/60 transition-colors">
                  {columns.map((col, idx) => (
                    <td key={idx} className={cn("px-3.5 py-2.5 align-middle", col.className)}>
                      {col.cell
                        ? col.cell(item)
                        : col.accessorKey
                        ? String(item[col.accessorKey] ?? "—")
                        : "—"}
                    </td>
                  ))}
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>

      {totalPages > 1 && (
        <div className="p-2.5 border-t border-[var(--border-subtle)] bg-[var(--bg-subtle)]/40 flex flex-wrap items-center justify-between gap-2 text-[11px] font-mono text-[#91A0B4]">
          <span>
            Showing {(page - 1) * pageSize + 1}–{Math.min(page * pageSize, sorted.length)} of{" "}
            {sorted.length}
          </span>
          <div className="flex items-center gap-1.5">
            <Button
              variant="secondary"
              size="sm"
              onClick={() => setCurrentPage((p) => Math.max(1, p - 1))}
              disabled={page === 1}
            >
              Prev
            </Button>
            <span className="px-2 text-[#ECF3FC] tabular-nums" aria-current="page">
              {page} / {totalPages}
            </span>
            <Button
              variant="secondary"
              size="sm"
              onClick={() => setCurrentPage((p) => Math.min(totalPages, p + 1))}
              disabled={page === totalPages}
            >
              Next
            </Button>
          </div>
        </div>
      )}
    </div>
  );
}

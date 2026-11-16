"use client";

import React, { useState } from "react";
import { cn } from "@/lib/utils";
import { ChevronDown, ChevronUp, Search } from "lucide-react";

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
  className?: string;
  emptyText?: string;
  pageSize?: number;
}

export function DataTable<T>({
  data,
  columns,
  keyExtractor,
  searchPlaceholder = "Search records...",
  searchFilter,
  className,
  emptyText = "No records found.",
  pageSize = 15,
}: DataTableProps<T>) {
  const [search, setSearch] = useState("");
  const [sortKey, setSortKey] = useState<keyof T | null>(null);
  const [sortAsc, setSortAsc] = useState(false);
  const [currentPage, setCurrentPage] = useState(1);

  // Filter
  const filtered = data.filter((item) => {
    if (!search.trim()) return true;
    if (searchFilter) return searchFilter(item, search.toLowerCase());
    return JSON.stringify(item).toLowerCase().includes(search.toLowerCase());
  });

  // Sort
  const sorted = [...filtered].sort((a, b) => {
    if (!sortKey) return 0;
    const aVal = a[sortKey];
    const bVal = b[sortKey];
    if (aVal === bVal) return 0;
    if (aVal === null || aVal === undefined) return 1;
    if (bVal === null || bVal === undefined) return -1;
    if (aVal < bVal) return sortAsc ? -1 : 1;
    return sortAsc ? 1 : -1;
  });

  // Paginate
  const totalPages = Math.max(1, Math.ceil(sorted.length / pageSize));
  const paginated = sorted.slice((currentPage - 1) * pageSize, currentPage * pageSize);

  const handleSort = (col: Column<T>) => {
    if (!col.sortable || !col.accessorKey) return;
    if (sortKey === col.accessorKey) {
      setSortAsc(!sortAsc);
    } else {
      setSortKey(col.accessorKey);
      setSortAsc(false);
    }
  };

  return (
    <div className={cn("glass-panel rounded-lg overflow-hidden flex flex-col", className)}>
      {searchFilter && (
        <div className="p-3 border-b border-slate-800/80 flex items-center gap-2">
          <Search className="w-4 h-4 text-slate-400" />
          <input
            type="text"
            value={search}
            onChange={(e) => {
              setSearch(e.target.value);
              setCurrentPage(1);
            }}
            placeholder={searchPlaceholder}
            className="w-full bg-slate-900/60 border border-slate-800 rounded px-2.5 py-1 text-xs font-mono text-slate-200 placeholder-slate-400 focus:outline-none focus:border-cyan-400/50"
          />
        </div>
      )}

      <div className="overflow-x-auto w-full">
        <table className="w-full text-left text-xs font-mono">
          <thead className="bg-slate-900/90 text-slate-400 border-b border-slate-800 uppercase tracking-wider text-[11px]">
            <tr>
              {columns.map((col, idx) => (
                <th
                  key={idx}
                  onClick={() => handleSort(col)}
                  className={cn(
                    "px-3.5 py-2.5 font-semibold select-none",
                    col.sortable ? "cursor-pointer hover:text-cyan-400" : "",
                    col.className
                  )}
                >
                  <div className="flex items-center gap-1">
                    <span>{col.header}</span>
                    {col.sortable && sortKey === col.accessorKey && (
                      sortAsc ? <ChevronUp className="w-3 h-3 text-cyan-400" /> : <ChevronDown className="w-3 h-3 text-cyan-400" />
                    )}
                  </div>
                </th>
              ))}
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-800/50 text-slate-300">
            {paginated.length === 0 ? (
              <tr>
                <td colSpan={columns.length} className="px-4 py-8 text-center text-slate-400">
                  {emptyText}
                </td>
              </tr>
            ) : (
              paginated.map((item) => (
                <tr
                  key={keyExtractor(item)}
                  className="hover:bg-slate-800/40 transition-colors"
                >
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
        <div className="p-2.5 border-t border-slate-800/80 flex items-center justify-between text-xs font-mono text-slate-400">
          <span>
            Showing {(currentPage - 1) * pageSize + 1} - {Math.min(currentPage * pageSize, sorted.length)} of {sorted.length}
          </span>
          <div className="flex items-center gap-1">
            <button
              onClick={() => setCurrentPage((p) => Math.max(1, p - 1))}
              disabled={currentPage === 1}
              className="px-2 py-0.5 rounded border border-slate-800 hover:border-cyan-400/50 disabled:opacity-40 disabled:hover:border-slate-800"
            >
              PREV
            </button>
            <span className="px-2 text-slate-200">
              {currentPage} / {totalPages}
            </span>
            <button
              onClick={() => setCurrentPage((p) => Math.min(totalPages, p + 1))}
              disabled={currentPage === totalPages}
              className="px-2 py-0.5 rounded border border-slate-800 hover:border-cyan-400/50 disabled:opacity-40 disabled:hover:border-slate-800"
            >
              NEXT
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

"use client";

import { useState, useRef, useEffect, useCallback } from "react";
import { ChevronDown, X, CheckCircle2 } from "lucide-react";
import { cn } from "@/lib/utils";

/* ── Common alleles per locus (IPD-IMGT/HLA frequency ≥ 1%) ─────────── */
export const HLA_SUGGESTIONS: Record<string, string[]> = {
  A: [
    "A*01:01","A*02:01","A*02:02","A*02:05","A*02:06","A*02:07",
    "A*02:11","A*03:01","A*11:01","A*11:02","A*23:01","A*24:02",
    "A*24:03","A*25:01","A*26:01","A*29:02","A*30:01","A*30:02",
    "A*31:01","A*32:01","A*33:01","A*33:03","A*34:01","A*36:01",
    "A*66:01","A*68:01","A*68:02","A*69:01","A*74:01","A*80:01",
  ],
  B: [
    "B*07:02","B*08:01","B*13:01","B*13:02","B*14:01","B*14:02",
    "B*15:01","B*15:02","B*15:03","B*18:01","B*27:02","B*27:03",
    "B*27:05","B*35:01","B*35:02","B*35:03","B*37:01","B*38:01",
    "B*39:01","B*40:01","B*40:02","B*41:01","B*44:02","B*44:03",
    "B*46:01","B*47:01","B*48:01","B*51:01","B*52:01","B*53:01",
    "B*54:01","B*55:01","B*56:01","B*57:01","B*57:03","B*58:01",
    "B*59:01","B*67:01","B*73:01","B*78:01",
  ],
  C: [
    "C*01:02","C*02:02","C*03:02","C*03:03","C*03:04","C*04:01",
    "C*04:03","C*05:01","C*06:02","C*07:01","C*07:02","C*07:04",
    "C*08:01","C*08:02","C*12:02","C*12:03","C*14:02","C*15:02",
    "C*16:01","C*17:01","C*18:01",
  ],
  DRB1: [
    "DRB1*01:01","DRB1*01:02","DRB1*01:03","DRB1*03:01","DRB1*03:02",
    "DRB1*04:01","DRB1*04:02","DRB1*04:03","DRB1*04:04","DRB1*04:05",
    "DRB1*04:07","DRB1*04:08","DRB1*07:01","DRB1*08:01","DRB1*08:02",
    "DRB1*08:03","DRB1*09:01","DRB1*10:01","DRB1*11:01","DRB1*11:02",
    "DRB1*11:03","DRB1*11:04","DRB1*12:01","DRB1*12:02","DRB1*13:01",
    "DRB1*13:02","DRB1*13:03","DRB1*13:04","DRB1*14:01","DRB1*14:02",
    "DRB1*14:03","DRB1*14:04","DRB1*15:01","DRB1*15:02","DRB1*15:03",
    "DRB1*16:01","DRB1*16:02",
  ],
  DQB1: [
    "DQB1*02:01","DQB1*02:02","DQB1*03:01","DQB1*03:02","DQB1*03:03",
    "DQB1*03:04","DQB1*04:01","DQB1*04:02","DQB1*05:01","DQB1*05:02",
    "DQB1*05:03","DQB1*06:01","DQB1*06:02","DQB1*06:03","DQB1*06:04",
  ],
};

interface HLAComboboxProps {
  locus: string;
  value: string;
  onChange: (v: string) => void;
  placeholder?: string;
  className?: string;
}

export function HLACombobox({
  locus,
  value,
  onChange,
  placeholder,
  className,
}: HLAComboboxProps) {
  const alleles = HLA_SUGGESTIONS[locus] ?? [];
  const [open, setOpen]   = useState(false);
  const [query, setQuery] = useState(value);
  const [hiIdx, setHiIdx] = useState(-1);
  const container = useRef<HTMLDivElement>(null);
  const inputRef  = useRef<HTMLInputElement>(null);
  const listRef   = useRef<HTMLUListElement>(null);

  // Sync external → local
  useEffect(() => { setQuery(value); }, [value]);

  // Close on outside click
  useEffect(() => {
    function handler(e: MouseEvent) {
      if (container.current && !container.current.contains(e.target as Node)) {
        setOpen(false);
        setQuery(value); // revert to last confirmed value
      }
    }
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, [value]);

  // Scroll highlighted item into view
  useEffect(() => {
    if (!listRef.current || hiIdx < 0) return;
    const el = listRef.current.children[hiIdx] as HTMLElement | undefined;
    el?.scrollIntoView({ block: "nearest" });
  }, [hiIdx]);

  const filtered = useCallback(() => {
    const q = query.toLowerCase().replace(/\s/g, "");
    if (!q) return alleles.slice(0, 10);
    return alleles
      .filter((a) => a.toLowerCase().replace(/[*:]/g, "").includes(q.replace(/[*:]/g, "")))
      .slice(0, 12);
  }, [query, alleles]);

  const suggestions = filtered();
  const isKnown = alleles.includes(value);

  function select(allele: string) {
    onChange(allele);
    setQuery(allele);
    setOpen(false);
    setHiIdx(-1);
  }

  function handleKeyDown(e: React.KeyboardEvent<HTMLInputElement>) {
    if (!open && (e.key === "ArrowDown" || e.key === "Enter")) {
      setOpen(true);
      return;
    }
    switch (e.key) {
      case "ArrowDown":
        setHiIdx((i) => Math.min(i + 1, suggestions.length - 1));
        e.preventDefault();
        break;
      case "ArrowUp":
        setHiIdx((i) => Math.max(i - 1, -1));
        e.preventDefault();
        break;
      case "Enter":
        if (hiIdx >= 0 && suggestions[hiIdx]) {
          select(suggestions[hiIdx]);
          e.preventDefault();
        }
        break;
      case "Escape":
        setOpen(false);
        setQuery(value);
        break;
      case "Tab":
        setOpen(false);
        break;
    }
  }

  function handleInputChange(e: React.ChangeEvent<HTMLInputElement>) {
    setQuery(e.target.value);
    setOpen(true);
    setHiIdx(-1);
    // If cleared, propagate empty up
    if (!e.target.value) onChange("");
  }

  /** Highlight matching substring */
  function highlight(allele: string, q: string) {
    if (!q) return <span>{allele}</span>;
    const clean = q.replace(/[*:]/g, "").toLowerCase();
    const idx = allele.toLowerCase().replace(/[*:]/g, "").indexOf(clean);
    if (idx < 0) return <span>{allele}</span>;
    // Map back to original string positions (brute-force for short strings)
    let ci = 0;
    const chars: { ch: string; bold: boolean }[] = [];
    for (let i = 0; i < allele.length; i++) {
      const isSkip = allele[i] === "*" || allele[i] === ":";
      chars.push({ ch: allele[i], bold: !isSkip && ci >= idx && ci < idx + clean.length });
      if (!isSkip) ci++;
    }
    return (
      <span>
        {chars.map(({ ch, bold }, i) =>
          bold ? <b key={i} className="text-navy">{ch}</b> : <span key={i}>{ch}</span>
        )}
      </span>
    );
  }

  return (
    <div ref={container} className={cn("relative", className)}>
      {/* Input */}
      <div className="relative">
        <input
          ref={inputRef}
          type="text"
          autoComplete="off"
          spellCheck={false}
          value={query}
          placeholder={placeholder ?? `${locus}*02:01`}
          onChange={handleInputChange}
          onFocus={() => setOpen(true)}
          onKeyDown={handleKeyDown}
          className={cn(
            "flex h-10 w-full rounded-lg border bg-background px-3 py-2 pr-8 text-sm font-mono ring-offset-background placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-0 transition-[border-color,box-shadow]",
            isKnown
              ? "border-green-400/70 focus-visible:ring-green-300"
              : value
              ? "border-amber-300/70 focus-visible:ring-amber-200"
              : "border-input"
          )}
        />
        {/* Right icon */}
        {value ? (
          <button
            type="button"
            tabIndex={-1}
            onClick={() => { onChange(""); setQuery(""); inputRef.current?.focus(); }}
            className="absolute right-2.5 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground transition-colors"
          >
            <X className="h-3.5 w-3.5" />
          </button>
        ) : (
          <ChevronDown className="pointer-events-none absolute right-2.5 top-1/2 -translate-y-1/2 h-3.5 w-3.5 text-muted-foreground" />
        )}
      </div>

      {/* Valid indicator */}
      {isKnown && (
        <span className="absolute -top-0.5 -right-0.5 text-green-500">
          <CheckCircle2 className="h-3 w-3" />
        </span>
      )}

      {/* Dropdown */}
      {open && suggestions.length > 0 && (
        <div className="absolute z-50 left-0 right-0 top-[calc(100%+4px)] rounded-lg border border-border bg-white shadow-lg overflow-hidden">
          <ul
            ref={listRef}
            className="max-h-48 overflow-y-auto scrollbar-thin py-1"
          >
            {suggestions.map((allele, i) => (
              <li
                key={allele}
                onMouseDown={(e) => { e.preventDefault(); select(allele); }}
                onMouseEnter={() => setHiIdx(i)}
                className={cn(
                  "px-3 py-1.5 text-sm font-mono cursor-pointer flex items-center justify-between",
                  i === hiIdx
                    ? "bg-navy text-white"
                    : allele === value
                    ? "bg-green-50 text-green-800"
                    : "hover:bg-muted"
                )}
              >
                {highlight(allele, query)}
                {allele === value && <CheckCircle2 className="h-3.5 w-3.5 shrink-0 ml-2 text-green-500" />}
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}

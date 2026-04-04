"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { FlaskConical } from "lucide-react";
import { GithubIcon } from "@/components/ui/github-icon";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

const NAV_LINKS = [
  { href: "/",        label: "Home"    },
  { href: "/predict", label: "Predict" },
  { href: "/about",   label: "About"   },
  { href: "/paper",   label: "Paper"   },
];

export default function Navbar() {
  const pathname = usePathname();

  return (
    <header className="sticky top-0 z-50 w-full border-b border-border/60 bg-white/90 backdrop-blur-md supports-[backdrop-filter]:bg-white/80">
      <div className="container flex h-16 max-w-7xl items-center justify-between px-6">
        {/* Logo */}
        <Link href="/" className="flex items-center gap-2.5 group">
          <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-navy text-white font-bold text-sm group-hover:bg-navy-700 transition-colors">
            C
          </div>
          <span className="font-semibold text-navy text-lg tracking-tight">
            CAPA
          </span>
          <span className="hidden sm:block text-xs text-muted-foreground font-normal mt-px">
            / Alloimmunity Prediction
          </span>
        </Link>

        {/* Nav links */}
        <nav className="hidden md:flex items-center gap-1">
          {NAV_LINKS.map(({ href, label }) => {
            const active = href === "/" ? pathname === "/" : pathname.startsWith(href);
            return (
              <Link
                key={href}
                href={href}
                className={cn(
                  "px-3.5 py-1.5 rounded-md text-sm font-medium transition-colors",
                  active
                    ? "text-navy bg-navy/5"
                    : "text-muted-foreground hover:text-navy hover:bg-navy/5"
                )}
              >
                {label}
              </Link>
            );
          })}
        </nav>

        {/* Right actions */}
        <div className="flex items-center gap-2">
          <Link
            href="https://github.com/capa-project/capa"
            target="_blank"
            rel="noopener noreferrer"
            className="hidden sm:flex items-center gap-1.5 text-sm text-muted-foreground hover:text-navy transition-colors px-2 py-1.5"
          >
            <GithubIcon className="h-4 w-4" />
            <span>GitHub</span>
          </Link>
          <Button asChild size="sm" variant="blush" className="gap-1.5">
            <Link href="/predict">
              <FlaskConical className="h-3.5 w-3.5" />
              Try it
            </Link>
          </Button>
        </div>
      </div>
    </header>
  );
}

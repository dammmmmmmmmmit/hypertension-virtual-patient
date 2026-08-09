"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

const LINKS = [
  { href: "/", label: "Demo", shortLabel: "Demo" },
  { href: "/architecture", label: "Architecture", shortLabel: "Arch." },
  { href: "/domain", label: "Domain", shortLabel: "Domain" },
  { href: "/data", label: "Data & methodology", shortLabel: "Data" },
];

// Nav stays neutral/indigo — acid is reserved for the one primary
// interactive element per view (ui_build_brief.md color discipline),
// which on most pages is something else (e.g. the Demo page's "Run
// simulation" button). The active link uses a quiet indigo underline,
// not acid, so it never competes with that rule.
export function SiteNav() {
  const pathname = usePathname();

  return (
    <header className="sticky top-0 z-20 border-b border-border bg-background/85 backdrop-blur">
      {/* flex-wrap is a safety net; the real fix for narrow viewports is
          shortLabel below sm — "Data & methodology" alone can approach a
          375px viewport's width, so a plain nowrap row would overflow. */}
      <nav className="mx-auto flex max-w-5xl flex-wrap items-center gap-x-1 gap-y-1.5 px-4 py-3 text-sm sm:px-6">
        <span className="mr-2 font-heading text-sm font-medium text-foreground sm:mr-4">
          VPDS
        </span>
        {LINKS.map((link) => {
          const active = pathname === link.href;
          return (
            <Link
              key={link.href}
              href={link.href}
              aria-current={active ? "page" : undefined}
              className={`rounded-md px-2.5 py-1.5 transition-colors sm:px-3 ${
                active
                  ? "bg-indigo/15 text-foreground"
                  : "text-muted-foreground hover:text-foreground"
              }`}
            >
              <span className="sm:hidden">{link.shortLabel}</span>
              <span className="hidden sm:inline">{link.label}</span>
            </Link>
          );
        })}
      </nav>
    </header>
  );
}

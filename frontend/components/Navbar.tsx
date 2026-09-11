"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

import { cx } from "@/lib/format";

const links = [
  { href: "/players", label: "Players" },
  { href: "/teams/904", label: "Team Intelligence" },
  { href: "/scouting", label: "Scouting" },
  { href: "/leaderboard", label: "Leaderboard" },
  { href: "/compare", label: "Compare" },
  { href: "/archetypes", label: "Archetypes" },
  { href: "/model", label: "Model" },
];

export function Navbar() {
  const pathname = usePathname();
  return (
    <header className="sticky top-0 z-50 border-b border-[var(--border)] bg-[#090d0cf2] backdrop-blur">
      <nav
        aria-label="Primary navigation"
        className="mx-auto flex max-w-7xl items-center gap-6 px-5 py-4 sm:px-8"
      >
        <Link href="/" className="mr-auto flex items-center gap-3 font-semibold tracking-tight">
          <span
            aria-hidden="true"
            className="grid size-9 place-items-center rounded-full border border-emerald-200/30 bg-emerald-300/10 text-sm font-black text-[var(--accent-strong)]"
          >
            FS
          </span>
          <span className="text-lg">FootyScout</span>
        </Link>
        <div className="flex items-center gap-1 overflow-x-auto">
          {links.map((link) => {
            const active = pathname === link.href || pathname.startsWith(`${link.href}/`);
            return (
              <Link
                key={link.href}
                href={link.href}
                aria-current={active ? "page" : undefined}
                className={cx(
                  "rounded-lg px-3 py-2 text-sm font-medium whitespace-nowrap transition-colors",
                  active
                    ? "bg-emerald-300/10 text-[var(--accent-strong)]"
                    : "text-[var(--muted)] hover:bg-white/5 hover:text-white",
                )}
              >
                {link.label}
              </Link>
            );
          })}
        </div>
      </nav>
    </header>
  );
}

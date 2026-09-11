import type { Metadata } from "next";

import { Navbar } from "@/components/Navbar";

import "./globals.css";

export const metadata: Metadata = {
  title: {
    default: "FootyScout — football scouting intelligence",
    template: "%s | FootyScout",
  },
  description:
    "FootyScout football scouting intelligence built from available StatsBomb event data.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>
        <Navbar />
        {children}
        <footer className="border-t border-[var(--border)] px-5 py-8 text-sm text-[var(--muted)] sm:px-8">
          <div className="mx-auto flex max-w-7xl flex-col gap-2 sm:flex-row sm:justify-between">
            <p>FootyScout · Football scouting and analytics</p>
            <p>Based on available StatsBomb 2023/24 Bundesliga event data.</p>
          </div>
        </footer>
      </body>
    </html>
  );
}

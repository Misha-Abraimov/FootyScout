import type { Metadata } from "next";

import { ArchetypeCatalogue } from "@/components/ArchetypeCatalogue";
import { PageHeader } from "@/components/PageHeader";
import { ErrorState } from "@/components/States";
import { api } from "@/lib/api";

export const dynamic = "force-dynamic";
export const metadata: Metadata = { title: "Player archetypes" };

export default async function ArchetypesPage() {
  const catalogue = await api.getArchetypes().catch(() => null);
  return (
    <main className="mx-auto min-h-screen max-w-6xl px-5 py-10 sm:px-8 sm:py-14">
      <PageHeader
        eyebrow="Player intelligence"
        title="Playing-style archetypes"
        description="Two data-derived outfield styles from FootyScout&apos;s six-dimensional, position-relative feature space. Neither style is better than the other."
      />
      <div className="mt-8">
        {catalogue ? <ArchetypeCatalogue catalogue={catalogue} /> : <ErrorState title="Archetypes unavailable" message="The archetype catalogue could not be loaded." />}
      </div>
    </main>
  );
}

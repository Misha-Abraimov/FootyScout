import type { Metadata } from "next";

import { AIScoutExperience } from "@/components/AIScoutExperience";
import { PageHeader } from "@/components/PageHeader";

export const metadata: Metadata = { title: "AI Scout" };

export default function AIScoutPage() {
  return (
    <main className="mx-auto min-h-screen max-w-5xl px-5 py-12 sm:px-8 sm:py-16">
      <PageHeader
        neutral
        eyebrow="Grounded scouting assistant"
        title="AI Scout"
        description="Ask about FootyScout player analytics, playing-style similarity, role fit, methodology, and current public context. Every answer is checked against its supplied evidence."
      />
      <AIScoutExperience />
    </main>
  );
}

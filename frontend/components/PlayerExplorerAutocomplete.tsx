"use client";

import { useRouter } from "next/navigation";

import { PlayerAutocomplete } from "@/components/PlayerAutocomplete";

export function PlayerExplorerAutocomplete({ initialValue }: { initialValue: string }) {
  const router = useRouter();
  return (
    <PlayerAutocomplete
      label="Player search"
      name="search"
      initialValue={initialValue}
      placeholder="e.g. Xhaka"
      className="min-w-0 md:col-span-2 lg:col-span-1"
      onSelect={(player) => router.push(`/players/${player.player_id}`)}
    />
  );
}

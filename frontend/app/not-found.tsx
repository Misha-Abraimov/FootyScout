import { EmptyState } from "@/components/States";

export default function NotFound() {
  return (
    <main className="mx-auto min-h-[70vh] max-w-7xl px-5 py-16 sm:px-8">
      <EmptyState
        title="That page is out of play"
        message="The player or page you requested could not be found."
        href="/players"
        action="Explore players"
      />
    </main>
  );
}

import { useEffect, useState } from "react";

type Props = {
  onRestart: () => void;
};

// Restart control (issue #13) — clears every node and returns the graph to its
// empty/landing state. Lives inside the Graph Info card's body (so it collapses
// away with the rest on mobile). Click once to arm ("Sure?"), click again to
// confirm, so the graph isn't nuked by accident. Disarms after a few seconds or
// when the pointer leaves. Styled to match the card's inline controls (the
// blocked-artist chips), not the standalone glass pills.
export function RestartButton({ onRestart }: Props) {
  const [armed, setArmed] = useState(false);

  useEffect(() => {
    if (!armed) return;
    const t = setTimeout(() => setArmed(false), 3000);
    return () => clearTimeout(t);
  }, [armed]);

  return (
    <button
      type="button"
      onClick={() => {
        if (armed) {
          onRestart();
          setArmed(false);
        } else {
          setArmed(true);
        }
      }}
      onMouseLeave={() => setArmed(false)}
      aria-label={armed ? "Confirm start over" : "Start over"}
      title={armed ? "Click again to clear the graph" : "Start over"}
      className={`flex w-full items-center justify-center gap-2 rounded-[12px] border px-3 py-2 text-xs font-medium transition ${
        armed
          ? "border-red-300 bg-red-50/70 text-red-500"
          : "border-[#d0d0d0] bg-white/60 text-[#5a5a5a] hover:border-[#a0a0a0] hover:bg-white"
      }`}
    >
      <svg width="13" height="13" viewBox="0 0 14 14" fill="none" aria-hidden>
        <path
          d="M11.5 5.5A4.75 4.75 0 1 0 12 8.5M11.5 1.5v4h-4"
          stroke="currentColor"
          strokeWidth="1.5"
          strokeLinecap="round"
          strokeLinejoin="round"
        />
      </svg>
      <span className="leading-none">{armed ? "Confirm Start Over" : "Start Over"}</span>
    </button>
  );
}

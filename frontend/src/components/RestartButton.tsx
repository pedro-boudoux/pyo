import { useEffect, useState } from "react";

type Props = {
  onRestart: () => void;
};

// Restart control (issue #13) — clears every node and returns the graph to its
// empty/landing state. Click once to arm ("Start over?"), click again to confirm,
// so the graph isn't nuked by accident. Disarms on its own after a few seconds or
// when the pointer leaves. Matches the glass visual language of ZoomControls.
export function RestartButton({ onRestart }: Props) {
  const [armed, setArmed] = useState(false);

  useEffect(() => {
    if (!armed) return;
    const t = setTimeout(() => setArmed(false), 3000);
    return () => clearTimeout(t);
  }, [armed]);

  return (
    <button
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
      className="relative overflow-hidden rounded-[25px] h-[40px] flex items-center justify-center gap-2 px-[14px] shadow-[0px_1px_4.1px_0px_rgba(0,0,0,0.25)] transition-all duration-200 hover:scale-[1.025] hover:shadow-[0px_6px_20px_rgba(0,0,0,0.18)]"
    >
      <div
        aria-hidden
        className="absolute inset-0 backdrop-blur-[2.5px] bg-white/75 rounded-[25px] pointer-events-none"
      />
      <div
        aria-hidden
        className="absolute inset-0 pointer-events-none rounded-[25px] shadow-[inset_0px_4px_4px_0px_rgba(255,255,255,0.25)]"
      />
      <span
        className={`relative flex items-center justify-center transition-colors ${
          armed ? "text-red-500" : "text-[#09090B]/60"
        }`}
      >
        <svg width="14" height="14" viewBox="0 0 14 14" fill="none">
          <path
            d="M11.5 5.5A4.75 4.75 0 1 0 12 8.5M11.5 1.5v4h-4"
            stroke="currentColor"
            strokeWidth="1.5"
            strokeLinecap="round"
            strokeLinejoin="round"
          />
        </svg>
      </span>
      <span
        className={`relative text-xs font-medium leading-none transition-colors ${
          armed ? "text-red-500" : "text-[#09090B]/60"
        }`}
      >
        {armed ? "Sure?" : "Start over"}
      </span>
    </button>
  );
}

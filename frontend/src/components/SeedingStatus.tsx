import { useEffect, useRef, useState } from "react";

// Static "..." and the equalizer bars / scrim live in
// index.css (`.loading-dots`, `.eq-bar`, `.seed-scrim`).

type Phase = "checking" | "warm" | "cold";

type Props = {
  phase: Phase;
  compact?: boolean;
  className?: string;
};

// Headline shown while the graph is being built. One is picked at random per
// seed and held for the whole build — no alternating.
const BUILDING_MESSAGES = [
  "Professionally vibe curating",
  "Getting ready to pyo (put you on)",
  "Freeing you from the shackles of bad song recommendations",
  "Predicting the next big thing",
  "Consulting my orb",
  "Avoiding the mainstream",
  "Making sure to stand out",
  "Finding your next favourite song",
  "Analyzing 808s",
];

// Cold-seed reassurance. The first line is the honest heads-up; once a seed
// drags on we cycle the rest every COLD_ROTATE_MS so the user feels kept-posted.
const COLD_MESSAGES = [
  "First time seeing this song, give us a minute this could take a while!",
  "It's not frozen, I swear!",
  "Yeah, I meant it when I said this'd take a while…",
  "Still here, still digging.",
  "Good things take time (this is one of 'em).",
  "The underground doesn't index itself.",
  "Almost worth the wait, promise.",
  "Fetching tags, doing math, being thorough.",
];

const COLD_ROTATE_MS = 20000;

// Per-bar timing — fixed (no Math.random in render) but irregular enough to read
// as an organic, dancing equalizer rather than a synchronized pulse.
const BAR_CFG = [
  { dur: 820, delay: 0 },
  { dur: 1180, delay: 130 },
  { dur: 700, delay: 270 },
  { dur: 1320, delay: 80 },
  { dur: 760, delay: 210 },
  { dur: 1080, delay: 340 },
  { dur: 900, delay: 170 },
];

function Equalizer({ compact }: { compact: boolean }) {
  const bars = compact ? 5 : 7;
  const height = compact ? 16 : 38;
  const width = compact ? 3 : 5;
  return (
    <div
      className="flex items-end justify-center"
      style={{ height, gap: compact ? 3 : 5 }}
      aria-hidden
    >
      {BAR_CFG.slice(0, bars).map((c, i) => (
        <span
          key={i}
          className={`eq-bar ${compact ? "" : "eq-bar--glow"}`}
          style={{
            width,
            height,
            animationDuration: `${c.dur}ms`,
            animationDelay: `${c.delay}ms`,
          }}
        />
      ))}
    </div>
  );
}

export function SeedingStatus({ phase, compact = false, className = "" }: Props) {
  // Headline: one random line, chosen once and held for the whole build.
  const [buildMessage] = useState(
    () => BUILDING_MESSAGES[Math.floor(Math.random() * BUILDING_MESSAGES.length)],
  );

  // Cold line: hold message 0 for the first window, then cycle 1..n every 15s.
  // The component unmounts when seeding ends, so state starts fresh per seed.
  const [coldIndex, setColdIndex] = useState(0);
  const startedAt = useRef(0);
  useEffect(() => {
    if (phase !== "cold") return;
    startedAt.current = Date.now();
    const id = setInterval(() => {
      const step = Math.floor((Date.now() - startedAt.current) / COLD_ROTATE_MS);
      // step 0 → first message, then 1, 2, … through the rest, looping.
      const idx = step === 0 ? 0 : 1 + ((step - 1) % (COLD_MESSAGES.length - 1));
      setColdIndex(idx);
    }, 1000);
    return () => clearInterval(id);
  }, [phase]);

  const label = phase === "checking" ? "Checking song" : buildMessage;

  // ── Compact: a small glass chip above the search bar over the light graph. ──
  if (compact) {
    return (
      <div className={`flex flex-col items-center gap-2 ${className}`}>
        <div className="relative overflow-hidden rounded-full shadow-[0px_1px_4.1px_0px_rgba(0,0,0,0.25)]">
          <div aria-hidden className="absolute inset-0 backdrop-blur-[2.5px] bg-white/75 pointer-events-none rounded-full" />
          <div aria-hidden className="absolute inset-0 pointer-events-none rounded-full shadow-[inset_0px_4px_4px_0px_rgba(255,255,255,0.25)]" />
          <div className="relative flex items-center gap-2.5 px-3.5 py-1.5 text-xs font-medium text-black">
            <Equalizer compact />
            <span className="loading-dots whitespace-nowrap">{label}</span>
          </div>
        </div>
        {phase === "cold" && (
          <div
            key={coldIndex}
            className="fade-up text-center text-[11px] font-medium text-black/80"
          >
            {COLD_MESSAGES[coldIndex]}
          </div>
        )}
      </div>
    );
  }

  // ── Full-screen builder: pill-less, glowing bars + display-font message on
  //    the dark scrim (the scrim itself is painted by the App overlay). ──
  return (
    <div className={`flex flex-col items-center gap-6 ${className}`}>
      <Equalizer compact={false} />
      <div className="flex flex-col items-center gap-2.5 text-center">
        <span className="loading-dots whitespace-nowrap font-display text-2xl leading-none text-black">
          {label}
        </span>
        {phase === "cold" && (
          <span
            key={coldIndex}
            className="fade-up max-w-[19rem] text-sm font-medium text-black/60"
          >
            {COLD_MESSAGES[coldIndex]}
          </span>
        )}
      </div>
    </div>
  );
}

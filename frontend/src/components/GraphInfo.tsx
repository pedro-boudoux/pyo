import { useEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { getDominantTags } from "../api";
import { useIsMobile } from "../hooks/useIsMobile";
import { RestartButton } from "./RestartButton";
import { unblockArtist, useBlockedArtists } from "../services/blacklist";

const TIP_WIDTH = 240;

type Tag = { tag: string; weight: number; count: number; share: number };

type Props = {
  nodeCount: number;
  avgListeners: number;
  trackIds: string[];
  onRestart: () => void;
};

const SKELETON_WIDTHS = [82, 64, 73, 50];

// Real tooltip: works on hover/focus (desktop) and tap (mobile). Rendered through
// a portal because the Graph Info panel is `overflow-hidden`, which would clip an
// in-flow tooltip; the native `title` attribute it replaced never showed on touch.
function InfoTip({ text }: { text: string }) {
  const [open, setOpen] = useState(false);
  const [pos, setPos] = useState<{ top: number; left: number } | null>(null);
  const btnRef = useRef<HTMLButtonElement>(null);

  const place = () => {
    const el = btnRef.current;
    if (!el) return;
    const r = el.getBoundingClientRect();
    const margin = 12;
    // Right-align to the icon, then clamp inside the viewport (panel hugs the
    // right edge, so the tooltip grows leftward).
    const left = Math.max(
      margin,
      Math.min(r.right - TIP_WIDTH, window.innerWidth - TIP_WIDTH - margin),
    );
    setPos({ top: r.bottom + 8, left });
  };

  const show = () => {
    place();
    setOpen(true);
  };
  const hide = () => setOpen(false);

  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && hide();
    const onDown = (e: PointerEvent) => {
      if (btnRef.current && !btnRef.current.contains(e.target as Node)) hide();
    };
    window.addEventListener("keydown", onKey);
    window.addEventListener("pointerdown", onDown);
    window.addEventListener("scroll", hide, true);
    return () => {
      window.removeEventListener("keydown", onKey);
      window.removeEventListener("pointerdown", onDown);
      window.removeEventListener("scroll", hide, true);
    };
  }, [open]);

  return (
    <>
      <button
        ref={btnRef}
        type="button"
        aria-label={text}
        onClick={(e) => {
          e.stopPropagation();
          open ? hide() : show();
        }}
        onMouseEnter={show}
        onMouseLeave={hide}
        onFocus={show}
        onBlur={hide}
        className="text-[#b0b0b0] hover:text-[#656565] transition-colors cursor-help leading-none"
      >
        <svg viewBox="0 0 24 24" className="w-3 h-3" fill="none" stroke="currentColor" strokeWidth={2} aria-hidden>
          <circle cx="12" cy="12" r="9" />
          <path d="M12 11v5" strokeLinecap="round" />
          <path d="M12 8h.01" strokeLinecap="round" />
        </svg>
      </button>
      {open &&
        pos &&
        createPortal(
          <div
            role="tooltip"
            style={{ top: pos.top, left: pos.left, width: TIP_WIDTH }}
            className="fixed z-50 rounded-lg bg-[#2b2b2b] px-3 py-2 text-[11px] leading-snug text-white shadow-lg pointer-events-none"
          >
            {text}
          </div>,
          document.body,
        )}
    </>
  );
}

function Stat({ label, value, tip }: { label: string; value: string | number; tip?: string }) {
  return (
    <div className="flex flex-col gap-1">
      <span className="font-display font-medium text-[#3f3f3f] text-xl leading-none tabular-nums">
        {value}
      </span>
      <div className="flex items-center gap-1">
        <span className="font-medium text-[#909090] text-[10px] uppercase tracking-wider leading-none">
          {label}
        </span>
        {tip && <InfoTip text={tip} />}
      </div>
    </div>
  );
}

// Compact 1-2 digit listener formatting — matches the per-node readout in
// SongNode. Returns "12.3k", "1.2M", or the raw integer for small counts.
function formatListeners(n: number): string {
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`;
  if (n >= 1_000) return `${(n / 1_000).toFixed(1)}k`;
  return String(n);
}

export function GraphInfo({ nodeCount, avgListeners, trackIds, onRestart }: Props) {
  const [tags, setTags] = useState<Tag[]>([]);
  const [loadingTags, setLoadingTags] = useState(false);
  const blocked = useBlockedArtists();
  const isMobile = useIsMobile();
  // On phones the panel starts collapsed to a compact chip so it doesn't eat
  // half the screen; tapping the header expands the full stats + tags.
  const [open, setOpen] = useState(false);
  const expanded = !isMobile || open;

  // Stable string key so the effect only fires when node IDs change,
  // not on every position tick (which creates a new trackIds reference).
  const idsKey = trackIds.join(",");
  const trackIdsRef = useRef(trackIds);
  trackIdsRef.current = trackIds;

  useEffect(() => {
    if (trackIdsRef.current.length === 0) {
      setTags([]);
      setLoadingTags(false);
      return;
    }
    setLoadingTags(true);
    const handle = setTimeout(async () => {
      try {
        const data = await getDominantTags(trackIdsRef.current, 5);
        setTags(data.tags);
      } catch {
        // silently ignore
      } finally {
        setLoadingTags(false);
      }
    }, 600);
    return () => clearTimeout(handle);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [idsKey]);

  // Bars are scaled relative to the strongest tag so the list reads as a ranking
  // even when the top share is modest; the % label still shows the true share.
  const maxShare = tags.reduce((m, t) => Math.max(m, t.share), 0) || 1;
  const showTagSection = loadingTags || tags.length > 0;

  return (
    <div className="relative overflow-hidden rounded-[15px] shadow-[0px_1px_4.1px_0px_rgba(0,0,0,0.25)]">
      <div aria-hidden className="absolute inset-0 backdrop-blur-[2.5px] bg-white/75 rounded-[15px] pointer-events-none" />
      <div aria-hidden className="absolute inset-0 pointer-events-none rounded-[15px] shadow-[inset_0px_4px_4px_0px_rgba(255,255,255,0.25)]" />
      <div className="relative flex flex-col p-5 w-[200px]">
        <button
          type="button"
          onClick={() => isMobile && setOpen((o) => !o)}
          aria-expanded={expanded}
          className={`flex w-full items-center gap-2 ${isMobile ? "cursor-pointer" : "cursor-default"}`}
        >
          <svg
            viewBox="0 0 24 24"
            className="w-3.5 h-3.5 text-[#656565] shrink-0"
            fill="currentColor"
            aria-hidden
          >
            <path d="M18.5 3a2.5 2.5 0 1 1-.912 4.828l-4.556 4.555a5.48 5.48 0 0 1 .936 3.714l2.624.787a2.5 2.5 0 1 1-.575 1.916l-2.623-.788a5.5 5.5 0 0 1-10.39-2.29L3 15.5l.004-.221a5.5 5.5 0 0 1 2.984-4.673L5.2 7.982a2.5 2.5 0 0 1-2.194-2.304L3 5.5l.005-.164a2.5 2.5 0 1 1 4.111 2.071l.787 2.625a5.48 5.48 0 0 1 3.714.936l4.555-4.556a2.5 2.5 0 0 1-.167-.748L16 5.5l.005-.164A2.5 2.5 0 0 1 18.5 3" />
          </svg>
          <p className="font-display font-medium text-[#656565] text-base leading-none whitespace-nowrap">Graph Info</p>
          {isMobile && (
            <svg
              viewBox="0 0 24 24"
              className={`ml-auto w-3.5 h-3.5 text-[#909090] shrink-0 transition-transform ${open ? "rotate-180" : ""}`}
              fill="none"
              stroke="currentColor"
              strokeWidth={2}
              aria-hidden
            >
              <path d="M6 9l6 6 6-6" strokeLinecap="round" strokeLinejoin="round" />
            </svg>
          )}
        </button>

        {/* Body — animates height via the grid 0fr→1fr trick so expanding on
            mobile slides open smoothly instead of snapping. Spacing lives inside
            the clipped wrapper so the collapsed chip keeps its exact height. */}
        <div
          className={`grid transition-[grid-template-rows] duration-300 ease-out ${
            expanded ? "grid-rows-[1fr]" : "grid-rows-[0fr]"
          }`}
        >
          <div className="overflow-hidden">
          <div className="flex flex-col gap-4 pt-4">
        <div className="flex items-stretch gap-4">
          <Stat label="Songs" value={nodeCount} />
          <div aria-hidden className="w-px self-stretch bg-[#656565]/15" />
          <Stat label="Avg Listeners" value={formatListeners(avgListeners)} tip="Average Last.fm listener count across the songs on this graph." />
        </div>

        {showTagSection && (
          <>
            <div aria-hidden className="h-px bg-[#656565]/15" />
            <div className="flex flex-col gap-3">
              <div className="flex items-center gap-1.5">
                <p className="font-medium text-[#909090] text-[10px] uppercase tracking-wider leading-none">
                  Dominant Tags
                </p>
                <InfoTip text=
                         "Tags come from Last.fm's community, listeners tag each track and artist. We blend the track + artist tags, weight them by how many people applied each, then average across every song on the graph. Any odd or unexpected tags come from Last.fm's data, not from pyo."
                />
              </div>

              {loadingTags ? (
                <ul className="flex flex-col gap-3">
                  {SKELETON_WIDTHS.map((w, i) => (
                    <li key={i} className="flex flex-col gap-1.5">
                      <div
                        className="h-2.5 rounded-full bg-[#656565]/20 animate-pulse"
                        style={{ width: `${w}%`, animationDelay: `${i * 80}ms` }}
                      />
                      <div
                        className="h-1.5 rounded-full bg-[#656565]/12 animate-pulse"
                        style={{ width: `${w * 0.9}%`, animationDelay: `${i * 80}ms` }}
                      />
                    </li>
                  ))}
                </ul>
              ) : (
                <ul className="flex flex-col gap-2.5">
                  {tags.map((t) => {
                    const pct = Math.round(t.share * 100);
                    return (
                      <li key={t.tag} className="flex flex-col gap-1">
                        <div className="flex items-baseline justify-between gap-2">
                          <span className="capitalize font-medium text-[#656565] text-xs leading-normal truncate">
                            {t.tag}
                          </span>
                          <span className="font-medium text-[#909090] text-[10px] leading-none tabular-nums shrink-0">
                            {pct}%
                          </span>
                        </div>
                        <div className="h-1.5 rounded-full bg-[#656565]/12 overflow-hidden">
                          <div
                            className="h-full rounded-full bg-[#4a90d9]/70 transition-[width] duration-500 ease-out"
                            style={{ width: `${Math.max((t.share / maxShare) * 100, 6)}%` }}
                          />
                        </div>
                      </li>
                    );
                  })}
                </ul>
              )}
            </div>
          </>
        )}

        {blocked.length > 0 && (
          <>
            <div aria-hidden className="h-px bg-[#656565]/15" />
            <div className="flex flex-col gap-3">
              <p className="font-medium text-[#909090] text-[10px] uppercase tracking-wider leading-none">
                Blocked Artists
              </p>
              <ul className="flex flex-wrap gap-1.5">
                {blocked.map((artist) => (
                  <li key={artist}>
                    <button
                      type="button"
                      onClick={() => unblockArtist(artist)}
                      title={`Unblock ${artist}`}
                      className="group flex items-center gap-1 max-w-[150px] rounded-full border border-[#d0d0d0] bg-white/60 pl-2.5 pr-1.5 py-1 text-[11px] font-medium text-[#5a5a5a] hover:border-[#a0a0a0] hover:bg-white transition"
                    >
                      <span className="truncate">{artist}</span>
                      <span aria-hidden className="text-[#a0a0a0] group-hover:text-[#3a3a3a] leading-none">×</span>
                    </button>
                  </li>
                ))}
              </ul>
              <p className="text-[10px] text-[#909090] leading-snug">
                Not recommended to you. Tap to unblock.
              </p>
            </div>
          </>
        )}

        {/* Restart — part of the body so it collapses with the card on mobile,
            sitting alongside the tag/blocked-artist controls (issue #13). */}
        <div aria-hidden className="h-px bg-[#656565]/15" />
        <RestartButton onRestart={onRestart} />
          </div>
          </div>
        </div>
      </div>
    </div>
  );
}
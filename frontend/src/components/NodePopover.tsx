import { useCallback, useEffect, useRef, useState } from "react";
import type { ExpansionMethod, ExpansionParams } from "../types";
import { DEFAULT_EXPANSION } from "../types";
import { getSongFeatures } from "../api";
import { getCachedSpotifyLink, prefetchSpotifyLink } from "../services/spotifyCache";

// How many of the track's top tags to show under the title.
const MAX_TAGS = 5;
// Tags rarely change for a track, so cache them across popover opens.
const tagCache = new Map<string, string[]>();

// Keep in sync with the .popover-out duration in index.css
const EXIT_MS = 150;
// Mobile bottom-sheet slide duration + how far you must drag the handle down
// before releasing dismisses the sheet (otherwise it snaps back).
const SHEET_MS = 280;
const DISMISS_THRESHOLD = 110;

const METHOD_LABELS: Record<ExpansionMethod, string> = {
  recommendations: "MMR",
  linear: "Linear",
  tree: "Tree",
};

const METHOD_DESCRIPTIONS: Record<ExpansionMethod, string> = {
  recommendations: "Balances similarity to the seed with diversity between picks, so you get varied but relevant music.",
  linear: "Flat ranked list of the seed's nearest neighbors, sorted by similarity.",
  tree: "BFS through the graph from the seed, branching outward to explore further connections.",
};

type SpotifyState =
    
  | { status: "loading" }
  | { status: "found"; url: string }
  | { status: "unavailable" };

type Props = {
  nodeLabel: string;
  /** The node's artist — used by the "Block artist" action. */
  artist?: string;
  trackId: string;
  isSeed: boolean;
  /** Top tags delivered with the node's expansion response (issue #22). When
   *  present the popover shows them immediately; otherwise it falls back to a
   *  /features fetch (e.g. the seed node, which carries no tags). */
  tags?: string[];
  loading: boolean;
  onExpand: (params: ExpansionParams) => void;
  onDelete: () => void;
  /** Block this artist everywhere (recommendations + pull from graph). */
  onBlockArtist?: () => void;
  onClose: () => void;
  initial?: ExpansionParams;
  /** Render as a full-width bottom sheet (mobile) instead of a floating card. */
  mobile?: boolean;
  /** Desktop only — initial top-left position of the floating card (px). */
  x?: number;
  y?: number;
};

export function NodePopover({
  nodeLabel,
  artist,
  trackId,
  isSeed,
  tags: tagsProp,
  loading,
  onExpand,
  onDelete,
  onBlockArtist,
  onClose,
  initial,
  mobile = false,
  x = 0,
  y = 0,
}: Props) {
  const [params, setParams] = useState<ExpansionParams>(initial ?? DEFAULT_EXPANSION);
  const [confirmingDelete, setConfirmingDelete] = useState(false);
  const [confirmingBlock, setConfirmingBlock] = useState(false);
  const [leaving, setLeaving] = useState(false);

  // ── Desktop: drag the card by its header to reposition it anywhere ──
  // Position lives here (not in the parent) so dragging only re-renders the
  // popover. Initial spot is clamped so the card opens fully on-screen.
  const [pos, setPos] = useState(() => ({
    x: Math.min(x, window.innerWidth - 320),
    y: Math.min(y, window.innerHeight - 400),
  }));
  const dragOffsetRef = useRef<{ x: number; y: number } | null>(null);

  const onHeaderPointerDown = useCallback(
    (e: React.PointerEvent) => {
      if (mobile) return;
      // Let the close button / Spotify link handle their own clicks.
      if ((e.target as HTMLElement).closest("a, button, input")) return;
      dragOffsetRef.current = { x: e.clientX - pos.x, y: e.clientY - pos.y };
      (e.currentTarget as HTMLElement).setPointerCapture(e.pointerId);
    },
    [mobile, pos.x, pos.y],
  );

  const onHeaderPointerMove = useCallback((e: React.PointerEvent) => {
    const off = dragOffsetRef.current;
    if (!off) return;
    // Clamp so the card stays on-screen (~300px wide) and the header is reachable.
    const nx = Math.min(Math.max(8, e.clientX - off.x), window.innerWidth - 308);
    const ny = Math.min(Math.max(8, e.clientY - off.y), window.innerHeight - 48);
    setPos({ x: nx, y: ny });
  }, []);

  const onHeaderPointerUp = useCallback((e: React.PointerEvent) => {
    if (!dragOffsetRef.current) return;
    dragOffsetRef.current = null;
    (e.currentTarget as HTMLElement).releasePointerCapture?.(e.pointerId);
  }, []);

  // ── Mobile bottom-sheet motion (slide in + swipe-the-handle-down to dismiss) ──
  const [sheetTransform, setSheetTransform] = useState("translateY(100%)");
  const [backdropOn, setBackdropOn] = useState(false);
  const [dragging, setDragging] = useState(false);
  const dragStartRef = useRef<number | null>(null);
  const dragYRef = useRef(0);

  // Slide up from the bottom on mount.
  useEffect(() => {
    if (!mobile) return;
    const id = requestAnimationFrame(() => {
      setSheetTransform("translateY(0px)");
      setBackdropOn(true);
    });
    return () => cancelAnimationFrame(id);
  }, [mobile]);
  // Links are prefetched + cached when nodes appear on the graph, so this is
  // usually a synchronous cache hit. The "loading" state only shows on the rare
  // miss (e.g. a node that appeared before its prefetch finished).
  const cached = getCachedSpotifyLink(trackId);
  const [spotify, setSpotify] = useState<SpotifyState>(
    cached === undefined
      ? { status: "loading" }
      : cached
        ? { status: "found", url: cached }
        : { status: "unavailable" },
  );

  useEffect(() => {
    if (cached !== undefined) return; // already resolved from cache
    let cancelled = false;
    setSpotify({ status: "loading" });
    prefetchSpotifyLink(trackId).then((url) => {
      if (cancelled) return;
      setSpotify(url ? { status: "found", url } : { status: "unavailable" });
    });
    return () => {
      cancelled = true;
    };
  }, [trackId, cached]);

  // The track's top tags, shown under the title. Preferred source is the tags
  // delivered with the expansion response (issue #22) — no round-trip. We only
  // fall back to a /features fetch for nodes that carry none (e.g. the seed node,
  // or older nodes created before this change). Cached after the first resolve.
  const propTags = tagsProp && tagsProp.length > 0 ? tagsProp.slice(0, MAX_TAGS) : null;
  const [tags, setTags] = useState<string[]>(
    () => propTags ?? tagCache.get(trackId) ?? [],
  );
  useEffect(() => {
    if (propTags) {
      tagCache.set(trackId, propTags);
      setTags(propTags);
      return;
    }
    if (tagCache.has(trackId)) {
      setTags(tagCache.get(trackId)!);
      return;
    }
    let cancelled = false;
    getSongFeatures(trackId)
      .then((f) => {
        const top = f.tags.slice(0, MAX_TAGS);
        tagCache.set(trackId, top);
        if (!cancelled) setTags(top);
      })
      .catch(() => {
        // non-fatal — just show no tags
      });
    return () => {
      cancelled = true;
    };
    // propTags is derived from tagsProp; trackId keys the cache/fetch.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [trackId, tagsProp]);

  // Play the exit animation, then let the parent unmount us. On mobile the sheet
  // slides back down (and the backdrop fades) before unmounting.
  const close = useCallback(() => {
    setLeaving((already) => {
      if (already) return already;
      if (mobile) {
        setSheetTransform("translateY(100%)");
        setBackdropOn(false);
        setTimeout(onClose, SHEET_MS);
      } else {
        setTimeout(onClose, EXIT_MS);
      }
      return true;
    });
  }, [onClose, mobile]);

  // Drag the grab handle down to dismiss; release short of the threshold snaps back.
  const onHandleTouchStart = useCallback((e: React.TouchEvent) => {
    dragStartRef.current = e.touches[0].clientY;
    dragYRef.current = 0;
    setDragging(true);
  }, []);

  const onHandleTouchMove = useCallback((e: React.TouchEvent) => {
    if (dragStartRef.current === null) return;
    const dy = Math.max(0, e.touches[0].clientY - dragStartRef.current);
    dragYRef.current = dy;
    setSheetTransform(`translateY(${dy}px)`);
  }, []);

  const onHandleTouchEnd = useCallback(() => {
    if (dragStartRef.current === null) return;
    dragStartRef.current = null;
    setDragging(false);
    if (dragYRef.current > DISMISS_THRESHOLD) {
      close();
    } else {
      setSheetTransform("translateY(0px)");
    }
  }, [close]);

  // Dismiss on Escape.
  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      if (e.key === "Escape") close();
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [close]);

  function update<K extends keyof ExpansionParams>(key: K, value: ExpansionParams[K]) {
    setParams((p) => ({ ...p, [key]: value }));
  }

  const card = (
    <div
      className={[
        "relative shadow-[0px_1px_4.1px_0px_rgba(0,0,0,0.25)]",
        mobile
          ? "w-full max-h-[85vh] overflow-y-auto rounded-t-2xl bg-white/95 backdrop-blur-md"
          : `w-[300px] overflow-hidden rounded-[15px] ${leaving ? "popover-out" : "popover-in"}`,
      ].join(" ")}
      style={
        mobile
          ? {
              transform: sheetTransform,
              transition: dragging ? "none" : `transform ${SHEET_MS}ms var(--ease-out)`,
            }
          : { transformOrigin: "top left" }
      }
    >
      {!mobile && (
        <>
          <div aria-hidden className="absolute inset-0 backdrop-blur-[4px] bg-white/[0.88] rounded-[15px] pointer-events-none" />
          <div aria-hidden className="absolute inset-0 pointer-events-none rounded-[15px] shadow-[inset_0px_4px_4px_0px_rgba(255,255,255,0.25)]" />
        </>
      )}
      {mobile && (
        <div
          onTouchStart={onHandleTouchStart}
          onTouchMove={onHandleTouchMove}
          onTouchEnd={onHandleTouchEnd}
          style={{ touchAction: "none" }}
          className="flex justify-center pt-3 pb-2 cursor-grab active:cursor-grabbing"
        >
          <div aria-hidden className="h-1.5 w-10 rounded-full bg-[#cfcfcf]" />
        </div>
      )}
      <div className={`relative p-4 text-sm text-[#3a3a3a] ${mobile ? "pb-[calc(1rem+env(safe-area-inset-bottom))]" : ""}`}>
        <div
          onPointerDown={onHeaderPointerDown}
          onPointerMove={onHeaderPointerMove}
          onPointerUp={onHeaderPointerUp}
          className={`flex items-start justify-between gap-2 mb-3 ${
            mobile ? "" : "cursor-grab select-none active:cursor-grabbing"
          }`}
        >
          <div className="min-w-0">
            <div className="text-[10px] uppercase tracking-widest text-[#8a8a8a] font-medium">
              Expand from
            </div>
            <div className="truncate font-semibold text-[#1a1a1a]">{nodeLabel}</div>
            {tags.length > 0 && (
              <div className="mt-1 text-[10px] leading-snug text-[#9a9a9a] capitalize">
                {tags.join(" · ")}
              </div>
            )}
            {spotify.status === "loading" && (
              <span className="mt-1 inline-flex items-center gap-1 text-xs font-medium text-[#8a8a8a]">
                <SpotifyIcon className="w-3.5 h-3.5 animate-pulse" />
                Finding on Spotify…
              </span>
            )}
            {spotify.status === "found" && (
              <a
                href={spotify.url}
                target="_blank"
                rel="noopener noreferrer"
                className="mt-1 inline-flex items-center gap-1 text-xs font-medium text-[#1DB954] hover:text-[#1aa34a] transition-colors"
              >
                <SpotifyIcon className="w-3.5 h-3.5" />
                Listen on Spotify
              </a>
            )}
            {spotify.status === "unavailable" && (
              <span className="mt-1 inline-flex items-center gap-1 text-xs font-medium text-[#b0b0b0]">
                <SpotifyIcon className="w-3.5 h-3.5" />
                Not on Spotify
              </span>
            )}
          </div>
          <button
            onClick={close}
            className="text-[#8a8a8a] hover:text-[#3a3a3a] px-1 leading-none transition-colors"
            aria-label="Close"
          >
            ×
          </button>
        </div>

        <Label>Method</Label>
        <div className="grid grid-cols-3 gap-1 mb-2">
          {(["recommendations", "linear", "tree"] as ExpansionMethod[]).map((m) => (
            <button
              key={m}
              onClick={() => update("method", m)}
              className={[
                "px-2 py-1.5 rounded-md text-xs border transition",
                params.method === m
                  ? "border-blue-400 text-blue-600 bg-blue-50/80 font-medium"
                  : "border-[#d0d0d0] text-[#8a8a8a] hover:text-[#3a3a3a] hover:border-[#a0a0a0]",
              ].join(" ")}
            >
              {METHOD_LABELS[m]}
            </button>
          ))}
        </div>
        <p className="text-[10px] text-[#8a8a8a] leading-snug mb-3">
          {METHOD_DESCRIPTIONS[params.method]}
        </p>

        <SliderRow
          label="Number of songs"
          value={params.k}
          min={1}
          max={10}
          step={1}
          suffix={String(params.k)}
          onChange={(v) => update("k", v)}
        />

        <SliderRow
          label="Min. similarity"
          value={params.minSimilarity}
          min={0}
          max={0.95}
          step={0.05}
          suffix={`${Math.round(params.minSimilarity * 100)}%`}
          onChange={(v) => update("minSimilarity", v)}
          hint={
            params.minSimilarity === 0
              ? "off — keep every match"
              : "drop matches below this"
          }
        />

        {params.method === "recommendations" && (
          <SliderRow
            label="Diversity (λ)"
            value={params.lambda}
            min={0}
            max={1}
            step={0.05}
            suffix={params.lambda.toFixed(2)}
            onChange={(v) => update("lambda", v)}
            hint={
              params.lambda > 0.7
                ? "close to seed"
                : params.lambda < 0.4
                  ? "very diverse"
                  : "balanced"
            }
          />
        )}

        {params.method === "tree" && (
          <SliderRow
            label="Max depth"
            value={params.maxDepth}
            min={1}
            max={5}
            step={1}
            suffix={String(params.maxDepth)}
            onChange={(v) => update("maxDepth", v)}
          />
        )}

        {(params.method === "linear" || params.method === "tree") && (
          <label className="flex items-center gap-2 mt-3 text-xs cursor-pointer select-none">
            <input
              type="checkbox"
              checked={params.niche}
              onChange={(e) => update("niche", e.target.checked)}
              className="accent-blue-500"
            />
            <span className="text-[#8a8a8a]">Niche mode (favor low listener counts)</span>
          </label>
        )}

        <label className="flex items-center gap-2 mt-3 text-xs cursor-pointer select-none">
          <input
            type="checkbox"
            checked={params.allowDuplicates}
            onChange={(e) => update("allowDuplicates", e.target.checked)}
            className="accent-blue-500"
          />
          <span className="text-[#8a8a8a]">Allow songs already in graph</span>
        </label>

        <button
          disabled={loading}
          onClick={() => onExpand(params)}
          className="mt-4 w-full bg-blue-500 text-white font-semibold rounded-xl py-2 text-sm disabled:opacity-40 disabled:cursor-not-allowed hover:bg-blue-600 transition-colors"
        >
          {loading ? "Expanding…" : "Expand"}
        </button>

        <div className="mt-3 pt-3 border-t border-[#e0e0e0]/60">
          <button
            disabled={loading}
            onClick={() => {
              if (confirmingDelete) {
                setLeaving(true);
                setTimeout(onDelete, EXIT_MS);
              } else setConfirmingDelete(true);
            }}
            className={[
              "w-full rounded-xl py-2 text-xs font-medium transition disabled:opacity-50 disabled:cursor-not-allowed",
              confirmingDelete
                ? "bg-red-500 text-white hover:bg-red-600"
                : "border border-[#d0d0d0] text-red-500 hover:bg-red-50/60 hover:border-red-300",
            ].join(" ")}
          >
            {confirmingDelete ? "Click again to remove" : "Remove from graph"}
          </button>
          {confirmingDelete && (
            <div className="text-[10px] text-[#8a8a8a] mt-1.5 text-center leading-snug">
              {isSeed
                ? "Removes this seed and everything branching from it."
                : "Also removes any songs left disconnected from a seed."}
            </div>
          )}

          {onBlockArtist && artist && (
            <div className="mt-2">
              <button
                disabled={loading}
                onClick={() => {
                  if (confirmingBlock) {
                    setLeaving(true);
                    setTimeout(onBlockArtist, EXIT_MS);
                  } else setConfirmingBlock(true);
                }}
                className={[
                  "w-full rounded-xl py-2 text-xs font-medium transition disabled:opacity-50 disabled:cursor-not-allowed",
                  confirmingBlock
                    ? "bg-[#3a3a3a] text-white hover:bg-black"
                    : "border border-[#d0d0d0] text-[#6a6a6a] hover:bg-[#f0f0f0] hover:border-[#a0a0a0]",
                ].join(" ")}
              >
                <span className="block truncate px-2">
                  {confirmingBlock ? `Click again to block ${artist}` : `Block ${artist}`}
                </span>
              </button>
              {confirmingBlock && (
                <div className="text-[10px] text-[#8a8a8a] mt-1.5 text-center leading-snug">
                  Removes every song by {artist} and stops recommending them. Unblock in Graph Info.
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );

  return (
    <>
      {mobile && (
        <div
          aria-hidden
          onClick={close}
          className="fixed inset-0 bg-black/30"
          style={{ opacity: backdropOn ? 1 : 0, transition: `opacity ${SHEET_MS}ms var(--ease-out)` }}
        />
      )}
      {mobile ? (
        card
      ) : (
        <div className="fixed z-30" style={{ left: pos.x, top: pos.y }}>
          {card}
        </div>
      )}
    </>
  );
}

function SpotifyIcon({ className }: { className?: string }) {
  return (
    <svg viewBox="0 0 24 24" className={className} fill="currentColor" aria-hidden>
      <path d="M12 0C5.4 0 0 5.4 0 12s5.4 12 12 12 12-5.4 12-12S18.66 0 12 0zm5.521 17.34c-.24.359-.66.48-1.021.24-2.82-1.74-6.36-2.101-10.561-1.141-.418.122-.779-.179-.899-.539-.12-.421.18-.78.54-.9 4.56-1.021 8.52-.6 11.64 1.32.42.18.479.659.301 1.02zm1.44-3.3c-.301.42-.841.6-1.262.3-3.239-1.98-8.159-2.58-11.939-1.38-.479.12-1.02-.12-1.14-.6-.12-.48.12-1.021.6-1.141C9.6 9.9 15 10.561 18.72 12.84c.361.181.54.78.241 1.2zm.12-3.36C15.24 8.4 8.82 8.16 5.16 9.301c-.6.179-1.2-.181-1.38-.721-.18-.601.18-1.2.72-1.381 4.26-1.26 11.28-1.02 15.721 1.621.539.3.719 1.02.419 1.56-.299.421-1.02.599-1.559.3z" />
    </svg>
  );
}

function Label({ children }: { children: React.ReactNode }) {
  return (
    <div className="text-[10px] uppercase tracking-widest font-medium text-[#8a8a8a] mb-1">
      {children}
    </div>
  );
}

function SliderRow({
  label,
  value,
  min,
  max,
  step,
  suffix,
  hint,
  onChange,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  suffix: string;
  hint?: string;
  onChange: (v: number) => void;
}) {
  return (
    <div className="mb-2">
      <div className="flex items-center justify-between mb-1">
        <Label>{label}</Label>
        <span className="font-mono text-xs tabular-nums text-[#3a3a3a]">{suffix}</span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        className="w-full accent-blue-500"
      />
      {hint && <div className="text-[10px] text-[#8a8a8a] mt-0.5">{hint}</div>}
    </div>
  );
}

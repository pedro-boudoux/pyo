import type { CSSProperties } from "react";

type Props = {
  /** Grid cell size in px (line spacing). */
  cellSize?: number;
  /** Grid line color. */
  lineColor?: string;
  /** Pixels/second the grid crawls diagonally. */
  speed?: number;
  className?: string;
};

/**
 * The animated grid that sits behind the graph / landing hero.
 *
 * A pure-CSS replacement for the old canvas: the grid is a tiled
 * `repeating-linear-gradient` and the motion is a single `transform: translate`
 * of exactly one cell, looped. That runs on the compositor (GPU) — there is no
 * `requestAnimationFrame`, no per-frame JS, and nothing competing with ReactFlow
 * or the d3 force sim on the main thread. `prefers-reduced-motion` pauses it via
 * the global rule in index.css, leaving a static grid (which, being seamless,
 * looks identical to any animation frame).
 */
export function GridBackground({
  cellSize = 44,
  lineColor = "rgba(0,0,0,0.03)",
  speed = 30,
  className = "",
}: Props) {
  // One cell of travel per (cellSize / speed) seconds keeps the on-screen crawl
  // rate constant regardless of cell size, and loops seamlessly (period = 1 cell).
  const durationSec = cellSize / speed;

  // The panning layer is inset by one cell on every side so that translating it a
  // full cell never exposes an uncovered edge.
  const style: CSSProperties = {
    position: "absolute",
    inset: `-${cellSize}px`,
    backgroundImage:
      `repeating-linear-gradient(0deg, ${lineColor} 0 1px, transparent 1px ${cellSize}px),` +
      `repeating-linear-gradient(90deg, ${lineColor} 0 1px, transparent 1px ${cellSize}px)`,
    backgroundSize: `${cellSize}px ${cellSize}px`,
    animationDuration: `${durationSec}s`,
    // Consumed by the grid-pan keyframe (index.css).
    ["--grid-cell" as string]: `${cellSize}px`,
  };

  return (
    <div className={`absolute inset-0 overflow-hidden ${className}`} aria-hidden>
      <div className="grid-pan" style={style} />
    </div>
  );
}

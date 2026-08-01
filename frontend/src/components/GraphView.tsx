import type { Edge, Node, NodeDragHandler, NodeMouseHandler, OnEdgesChange, OnNodesChange } from "reactflow";
import { Graph, type GraphHandle } from "./Graph";
import { GraphInfo } from "./GraphInfo";
import { SearchBar } from "./SearchBar";
import { NoticeToast } from "./NoticeToast";
import { SeedingStatus } from "./SeedingStatus";
import { SpotifyExportButton } from "./SpotifyExportButton";
import type { SongNodeData } from "./SongNode";
import type { SongSearchResult } from "../types";

type SeedingPhase = null | "checking" | "warm" | "cold";

// Spotify blocks playlist creation for Development-mode apps (403 Forbidden).
// The integration code is complete and correct — it just needs Extended Quota
// approval. Flip VITE_SPOTIFY_EXPORT_ENABLED=true to re-enable the button once
// that's granted. See GitHub issue #5.
const SPOTIFY_EXPORT_ENABLED =
  import.meta.env.VITE_SPOTIFY_EXPORT_ENABLED === "true";

export type GraphViewProps = {
  nodes: Node<SongNodeData>[];
  edges: Edge[];
  onNodesChange: OnNodesChange;
  onEdgesChange: OnEdgesChange;
  onNodeClick: NodeMouseHandler;
  onPaneClick: () => void;
  onNodeDragStart?: NodeDragHandler;
  onNodeDrag?: NodeDragHandler;
  onNodeDragStop?: NodeDragHandler;
  onSeed: (song: SongSearchResult) => void;
  seedingPhase: SeedingPhase;
  /** Transient info message shown right above the search bar (e.g. expansion
   *  result counts — issue #27). Null hides it. */
  notice: string | null;
  onDismissNotice: () => void;
  trackIds: string[];
  /** Average Last.fm listener count across the graph's non-seed nodes. */
  avgListeners: number;
  graphRef: React.RefObject<GraphHandle | null>;
  onRestart: () => void;
};

export function GraphView({
  nodes,
  edges,
  onNodesChange,
  onEdgesChange,
  onNodeClick,
  onPaneClick,
  onNodeDragStart,
  onNodeDrag,
  onNodeDragStop,
  onSeed,
  seedingPhase,
  notice,
  onDismissNotice,
  trackIds,
  avgListeners,
  graphRef,
  onRestart,
}: GraphViewProps) {
  return (
    <>
      {/* Full-screen graph canvas */}
      <div className="absolute inset-0">
        <Graph
          ref={graphRef}
          nodes={nodes}
          edges={edges}
          onNodesChange={onNodesChange}
          onEdgesChange={onEdgesChange}
          onNodeClick={onNodeClick}
          onPaneClick={onPaneClick}
          onNodeDragStart={onNodeDragStart}
          onNodeDrag={onNodeDrag}
          onNodeDragStop={onNodeDragStop}
        />
      </div>

      {/* Credit card — top-left (condensed to pyo + GitHub icon on mobile) */}
      <div className="absolute left-3 top-3 sm:left-5 sm:top-[29px] z-10">
        <div className="relative overflow-hidden rounded-[15px] shadow-[0px_1px_4.1px_0px_rgba(0,0,0,0.25)]">
          <div aria-hidden className="absolute inset-0 backdrop-blur-[2.5px] bg-white/75 rounded-[15px] pointer-events-none" />
          <div aria-hidden className="absolute inset-0 pointer-events-none rounded-[15px] shadow-[inset_0px_4px_4px_0px_rgba(255,255,255,0.25)]" />
          <div className="relative flex flex-row items-center gap-2 p-5 sm:flex-col sm:items-start sm:gap-[10px]">
            <p className="font-display font-medium text-[#656565] text-base leading-none">pyo</p>
            <p className="hidden sm:block font-medium text-[#656565] text-xs leading-none">by pedro boudoux</p>
            <a
              href="https://github.com/pedro-boudoux/pyo"
              target="_blank"
              rel="noopener noreferrer"
              aria-label="github.com/pedro-boudoux"
              className="flex items-center gap-1.5 font-medium text-[#4a90d9] text-xs leading-none hover:underline"
            >
              <svg viewBox="0 0 16 16" width="14" height="14" className="sm:w-3 sm:h-3" fill="currentColor" aria-hidden>
                <path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z" />
              </svg>
              <span className="hidden sm:inline">github.com/pedro-boudoux/pyo</span>
            </a>
          </div>
        </div>
      </div>

      {/* Graph info / control — top-right. Restart lives inside the card body. */}
      <div className="absolute right-3 top-3 sm:right-5 sm:top-[29px] z-10">
        <GraphInfo nodeCount={nodes.length} avgListeners={avgListeners} trackIds={trackIds} onRestart={onRestart} />
      </div>

      {/* Search bar + seeding status + notices — bottom-center, stacked above
          the search bar exactly like the seeding spinner (issue #27). */}
      <div className="absolute bottom-4 sm:bottom-8 left-1/2 -translate-x-1/2 z-10 w-[430px] max-w-[calc(100%-24px)] sm:max-w-[calc(100%-80px)]">
        <NoticeToast message={notice} onDismiss={onDismissNotice} />
        {seedingPhase && (
          <div className="mb-2 flex justify-center">
            <SeedingStatus phase={seedingPhase} compact />
          </div>
        )}
        <SearchBar
          onPick={onSeed}
          placeholder="Start typing to add more sources to the graph."
          placeholderShort="Add more sources…"
          dropUp
          disabled={!!seedingPhase}
        />
      </div>

      {/* Spotify export — bottom-right (gated until Extended Quota is approved) */}
      {SPOTIFY_EXPORT_ENABLED && (
        <div className="absolute bottom-8 right-5 z-10">
          <SpotifyExportButton
            songs={nodes.map((n) => ({ name: n.data.name, artist: n.data.artist }))}
          />
        </div>
      )}
    </>
  );
}

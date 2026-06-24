import { useCallback, useEffect, useRef, useState } from "react";
import { exchangeCodeForToken } from "./services/spotify";
import { GridBackground } from "./components/GridBackground";
import {
  addEdge,
  useEdgesState,
  useNodesState,
  type Edge,
  type Node,
  type NodeMouseHandler,
} from "reactflow";
import { GraphView } from "./components/GraphView";
import { Landing } from "./components/Landing";
import { NodePopover } from "./components/NodePopover";
import { SeedingStatus } from "./components/SeedingStatus";
import { type SongNodeData } from "./components/SongNode";
import { type GraphHandle } from "./components/Graph";
import { expandFromTrack, getSongStatus, seedSong } from "./api";
import { prefetchSpotifyLink } from "./services/spotifyCache";
import { blockArtist, getBlockedArtists, isArtistBlocked } from "./services/blacklist";
import { useGraphSim } from "./hooks/useGraphSim";
import { useIsMobile } from "./hooks/useIsMobile";
import type { ExpansionParams, SongSearchResult } from "./types";

type SeedingPhase = null | "checking" | "warm" | "cold";

type Vec = { x: number; y: number };

const ARC_RADIUS = 260;
const STAGGER_MS = 55;

function arcAround(count: number, parentPos: Vec): Vec[] {
  const arcStart = -Math.PI * 0.85;
  const arcEnd = Math.PI * 0.85;
  return Array.from({ length: count }, (_, i) => {
    const t = count === 1 ? 0.5 : i / (count - 1);
    const angle = arcStart + (arcEnd - arcStart) * t;
    return {
      x: parentPos.x + ARC_RADIUS * Math.sin(angle),
      y: parentPos.y + ARC_RADIUS * (0.4 + 0.6 * Math.cos(angle)),
    };
  });
}

// Given an initial set of node ids to remove, also remove any node left
// disconnected from every surviving seed (orphan pruning). Shared by node
// deletion and artist-blocking. Returns the full set of ids to remove.
function withDisconnected(
  nodes: Node<SongNodeData>[],
  edges: Edge[],
  initial: Set<string>,
): Set<string> {
  const adjacency = new Map<string, string[]>();
  for (const e of edges) {
    if (!e.source || !e.target) continue;
    if (initial.has(e.source) || initial.has(e.target)) continue;
    const list = adjacency.get(e.source) ?? [];
    list.push(e.target);
    adjacency.set(e.source, list);
  }
  const seeds = nodes.filter((n) => !initial.has(n.id) && n.data.isSeed).map((n) => n.id);
  const reachable = new Set<string>(seeds);
  const queue = [...seeds];
  while (queue.length) {
    const cur = queue.shift()!;
    for (const next of adjacency.get(cur) ?? []) {
      if (!reachable.has(next)) { reachable.add(next); queue.push(next); }
    }
  }
  const removed = new Set<string>(initial);
  for (const n of nodes) {
    if (!initial.has(n.id) && !reachable.has(n.id)) removed.add(n.id);
  }
  return removed;
}

type PopoverState = { nodeId: string; label: string; artist: string; isSeed: boolean; tags?: string[]; x: number; y: number };

// Detect whether we're running inside the Spotify OAuth popup
const spotifyCallbackCode = new URLSearchParams(window.location.search).get("code");
const isSpotifyPopup = !!(spotifyCallbackCode && window.opener);

// Module-level guard: an auth code is single-use, but React StrictMode
// double-invokes effects in dev. This ensures we exchange the code exactly once.
let spotifyExchangeStarted = false;

export default function App() {
  const [nodes, setNodes, onNodesChange] = useNodesState<SongNodeData>([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState([]);
  const [popover, setPopover] = useState<PopoverState | null>(null);
  const [loading, setLoading] = useState(false);
  const [seedingPhase, setSeedingPhase] = useState<SeedingPhase>(null);
  const [error, setError] = useState<string | null>(null);
  // Informational message, distinct from a hard `error`: e.g. a seed landed fine
  // but turned up zero underground neighbours. Rendered as a neutral toast, not
  // red — the seed node still appears, this just explains the lone node.
  const [notice, setNotice] = useState<string | null>(null);

  const graphRef = useRef<GraphHandle>(null);
  const isMobile = useIsMobile();

  // Prefetch + cache each track's Spotify link as nodes appear on the graph, so
  // opening a node's popover is instant. Keyed on the *set* of node ids (sorted)
  // so this only fires when nodes are added/removed, not while dragging.
  const nodeIdKey = nodes.map((n) => n.id).sort().join(",");
  useEffect(() => {
    if (!nodeIdKey) return;
    for (const id of nodeIdKey.split(",")) prefetchSpotifyLink(id);
  }, [nodeIdKey]);

  // Handle Spotify OAuth callback — runs in the popup window
  useEffect(() => {
    const code = new URLSearchParams(window.location.search).get("code");
    if (!code) return;
    if (spotifyExchangeStarted) return; // guard StrictMode double-invoke
    spotifyExchangeStarted = true;

    if (window.opener) {
      exchangeCodeForToken(code)
        .then(() => {
          (window.opener as Window).postMessage({ type: "spotify_auth_done" }, "*");
        })
        .catch((err: Error) => {
          (window.opener as Window).postMessage(
            { type: "spotify_auth_error", error: err.message },
            "*",
          );
        })
        .finally(() => window.close());
    } else {
      // Fallback: full-page redirect (popup was blocked) — exchange and clean the URL
      exchangeCodeForToken(code)
        .catch(() => {})
        .finally(() => window.history.replaceState({}, "", window.location.pathname));
    }
  }, []);

  const { simNodesRef, syncSimulation, removeNodesFromSim, handleNodeDragStart, handleNodeDrag, handleNodeDragStop } =
    useGraphSim(setNodes);

  // ── Seed a song ──────────────────────────────────────────────────────────────

  const handleSeed = useCallback(
    async (song: SongSearchResult) => {
      setSeedingPhase("checking");
      setError(null);
      setNotice(null);
      try {
        let cached = false;
        try {
          const status = await getSongStatus(song.track_id);
          cached = status.cached;
        } catch {
          // non-fatal
        }
        setSeedingPhase(cached ? "warm" : "cold");

        const isFirstSeed = simNodesRef.current.size === 0;

        await seedSong(song.track_id);
        const initialChildren = await expandFromTrack(song.track_id, "recommendations", {
          k: 8, lambda: 0.7, niche: false, maxDepth: 3, excludeIds: [],
          excludeArtists: getBlockedArtists(),
        });

        // Seed landed but the backend found nothing similar underground for it.
        // We still place the seed node below; surface a friendly heads-up so the
        // lone node doesn't read as a silent failure. Not a hard `error`.
        if (initialChildren.length === 0) {
          setNotice(
            `Couldn't dig up any underground tracks near "${song.name}" — it's a little too off the map. Try seeding something else, or expand from here later.`,
          );
        }

        let seedPos: Vec;
        if (isFirstSeed) {
          seedPos = { x: 0, y: 0 };
        } else {
          const existingSim = simNodesRef.current.get(song.track_id);
          if (existingSim) {
            seedPos = { x: existingSim.x ?? 0, y: existingSim.y ?? 0 };
          } else {
            const maxX = Math.max(...Array.from(simNodesRef.current.values()).map((n) => n.x ?? 0), 0);
            seedPos = { x: maxX + 600, y: 0 };
          }
        }

        const childPositions = arcAround(initialChildren.length, seedPos);

        const seedNode: Node<SongNodeData> = {
          id: song.track_id,
          type: "song",
          position: seedPos,
          data: { name: song.name, artist: song.artist, image: song.image, isSeed: true },
        };

        const newChildNodes: Node<SongNodeData>[] = initialChildren
          .filter((c) => !simNodesRef.current.has(c.track_id))
          .map((c, i) => ({
            id: c.track_id,
            type: "song",
            position: childPositions[i],
            data: { name: c.name, artist: c.artist, image: c.image, isSeed: false, similarity: c.similarity, listeners: c.listeners, tags: c.tags },
          }));

        const newEdges: Edge[] = initialChildren.map((c) => ({
          id: `${song.track_id}->${c.track_id}`,
          source: song.track_id,
          target: c.track_id,
        }));

        // Seed appears immediately
        if (isFirstSeed) {
          setNodes([seedNode]);
          setEdges([]);
          syncSimulation([{ id: song.track_id, isSeed: true, x: seedPos.x, y: seedPos.y }], [], true);
        } else {
          setNodes((nds) => {
            const base = nds.some((n) => n.id === song.track_id)
              ? nds.map((n) => n.id === song.track_id ? { ...n, data: { ...n.data, isSeed: true } } : n)
              : [...nds, seedNode];
            return base;
          });
          if (!simNodesRef.current.has(song.track_id)) {
            syncSimulation([{ id: song.track_id, isSeed: true, x: seedPos.x, y: seedPos.y }], [], false);
          } else {
            const sn = simNodesRef.current.get(song.track_id)!;
            sn.fx = seedPos.x;
            sn.fy = seedPos.y;
          }
        }

        // Children stagger in one by one
        newChildNodes.forEach((node, i) => {
          const edge = newEdges.find((e) => e.target === node.id);
          setTimeout(() => {
            setNodes((nds) => [...nds, node]);
            if (edge) setEdges((eds) => addEdge(edge, eds));
            syncSimulation(
              [{ id: node.id, isSeed: false, x: node.position.x, y: node.position.y }],
              edge ? [{ source: String(edge.source), target: node.id }] : [],
              false,
            );
          }, STAGGER_MS * (i + 1));
        });

        // Fit the view once children have staggered in. Follow-up seeds always
        // re-fit; the very first seed only re-fits on mobile, where the default
        // zoom hugs the lone seed node and the surrounding arc spills offscreen.
        if (!isFirstSeed || isMobile) {
          const settleDelay = STAGGER_MS * (newChildNodes.length + 3);
          setTimeout(() => graphRef.current?.fitView(), settleDelay);
          // The force layout keeps spreading nodes for ~1s after they appear, so
          // on first load fit again once it settles to guarantee all songs fit.
          if (isFirstSeed) {
            setTimeout(() => graphRef.current?.fitView(), settleDelay + 750);
          }
        }
      } catch (err) {
        setError(err instanceof Error ? err.message : "Failed to seed song");
      } finally {
        setSeedingPhase(null);
      }
    },
    [setNodes, setEdges, syncSimulation, simNodesRef, isMobile],
  );

  // ── Expand from a node ───────────────────────────────────────────────────────

  const handleExpand = useCallback(
    async (params: ExpansionParams) => {
      if (!popover) return;
      setLoading(true);
      setError(null);
      try {
        const parentId = popover.nodeId;
        const knownIds = Array.from(simNodesRef.current.keys());
        const excludeIds = params.allowDuplicates ? [] : knownIds.filter((id) => id !== parentId);
        const fetched = await expandFromTrack(parentId, params.method, {
          k: params.k, lambda: params.lambda, niche: params.niche,
          maxDepth: params.maxDepth, excludeIds,
          excludeArtists: getBlockedArtists(),
        });
        const children =
          params.minSimilarity > 0
            ? fetched.filter((c) => c.similarity >= params.minSimilarity)
            : fetched;

        const parentSim = simNodesRef.current.get(parentId);
        const parentPos: Vec = { x: parentSim?.x ?? 0, y: parentSim?.y ?? 0 };

        const newChildren = children.filter((c) => !simNodesRef.current.has(c.track_id));
        const initialPositions = arcAround(newChildren.length, parentPos);

        const newNodes: Node<SongNodeData>[] = newChildren.map((c, i) => ({
          id: c.track_id,
          type: "song",
          position: initialPositions[i],
          data: { name: c.name, artist: c.artist, image: c.image, isSeed: false, similarity: c.similarity, listeners: c.listeners, tags: c.tags },
        }));

        const newEdges: Edge[] = children.map((c) => ({
          id: `${parentId}->${c.track_id}`,
          source: parentId,
          target: c.track_id,
        }));

        // Edges to already-present nodes go in immediately
        const edgesForExisting = newEdges.filter((e) => !newNodes.some((n) => n.id === e.target));
        if (edgesForExisting.length > 0) {
          setEdges((eds) => {
            let next = eds;
            for (const e of edgesForExisting) {
              if (!next.some((ex) => ex.id === e.id)) next = addEdge(e, next);
            }
            return next;
          });
        }

        // New nodes stagger in one by one
        newNodes.forEach((node, i) => {
          const edge = newEdges.find((e) => e.target === node.id);
          setTimeout(() => {
            setNodes((nds) => [...nds, node]);
            if (edge) setEdges((eds) => addEdge(edge, eds));
            syncSimulation(
              [{ id: node.id, isSeed: false, x: node.position.x, y: node.position.y }],
              edge ? [{ source: String(edge.source), target: node.id }] : [],
              false,
            );
          }, STAGGER_MS * i);
        });

        setPopover(null);
      } catch (err) {
        setError(err instanceof Error ? err.message : "Failed to expand");
      } finally {
        setLoading(false);
      }
    },
    [popover, setNodes, setEdges, syncSimulation, simNodesRef],
  );

  // ── Delete a node ────────────────────────────────────────────────────────────

  const removeNodeSet = useCallback(
    (initial: Set<string>) => {
      if (initial.size === 0) return;
      const removed = withDisconnected(nodes, edges, initial);
      setNodes((nds) => nds.filter((n) => !removed.has(n.id)));
      setEdges((eds) => eds.filter((e) => !removed.has(e.source!) && !removed.has(e.target!)));
      removeNodesFromSim(removed);
    },
    [nodes, edges, setNodes, setEdges, removeNodesFromSim],
  );

  const handleDeleteNode = useCallback(
    (nodeId: string) => {
      removeNodeSet(new Set([nodeId]));
      setPopover(null);
    },
    [removeNodeSet],
  );

  // Block an artist (persists to the user's local blacklist) and immediately pull
  // every node crediting them off the graph, along with anything left orphaned.
  // The block is also sent to the backend on future expansions via excludeArtists.
  const handleBlockArtist = useCallback(
    (artist: string) => {
      blockArtist(artist);
      const matching = new Set(
        nodes.filter((n) => isArtistBlocked(n.data.artist, [artist])).map((n) => n.id),
      );
      removeNodeSet(matching);
      setPopover(null);
    },
    [nodes, removeNodeSet],
  );

  // Restart (issue #13) — wipe every node/edge and return to the empty landing
  // state, exactly as it was before the first seed. Clears the physics sim through
  // its own bookkeeping (removeNodesFromSim) so no stale nodes leak in, and resets
  // the popover/error/seeding state.
  const handleRestart = useCallback(() => {
    removeNodesFromSim(new Set(simNodesRef.current.keys()));
    setNodes([]);
    setEdges([]);
    setPopover(null);
    setError(null);
    setSeedingPhase(null);
    setLoading(false);
  }, [removeNodesFromSim, simNodesRef, setNodes, setEdges]);

  const handleNodeClick: NodeMouseHandler = useCallback((event, node) => {
    const data = node.data as SongNodeData;
    setPopover({
      nodeId: node.id,
      label: `${data.name} — ${data.artist}`,
      artist: data.artist,
      isSeed: data.isSeed,
      tags: data.tags,
      x: event.clientX,
      y: event.clientY,
    });
  }, []);

  const hasGraph = nodes.length > 0;

  if (isSpotifyPopup) {
    return (
      <div className="h-full w-full flex items-center justify-center bg-[#FAFAFA] text-sm text-black/50 font-medium">
        Connecting to Spotify…
      </div>
    );
  }

  return (
    <div className="h-full w-full relative overflow-hidden bg-[#FAFAFA]">
      {hasGraph ? (
        <>
          <GridBackground />
          <GraphView
            nodes={nodes}
            edges={edges}
            onNodesChange={onNodesChange}
            onEdgesChange={onEdgesChange}
            onNodeClick={handleNodeClick}
            onPaneClick={() => setPopover(null)}
            onNodeDragStart={handleNodeDragStart}
            onNodeDrag={handleNodeDrag}
            onNodeDragStop={handleNodeDragStop}
            onSeed={handleSeed}
            seedingPhase={seedingPhase}
            trackIds={nodes.map((n) => n.id)}
            edgeCount={edges.length}
            graphRef={graphRef}
            onRestart={handleRestart}
          />
        </>
      ) : (
        <Landing onPick={handleSeed} disabled={!!seedingPhase} />
      )}

      {/* Full-screen builder overlay while seeding from the hero page */}
      {!hasGraph && seedingPhase && (
        <div className="absolute inset-0 z-50 flex items-center justify-center">
          <div aria-hidden className="seed-scrim absolute inset-0 backdrop-blur-md pointer-events-none" />
          <div className="relative fade-up">
            <SeedingStatus phase={seedingPhase} />
          </div>
        </div>
      )}

      {popover && (
        isMobile ? (
          <div className="fixed inset-0 z-30 flex items-end">
            <NodePopover
              key={popover.nodeId}
              nodeLabel={popover.label}
              artist={popover.artist}
              trackId={popover.nodeId}
              isSeed={popover.isSeed}
              tags={popover.tags}
              loading={loading}
              onExpand={handleExpand}
              onDelete={() => handleDeleteNode(popover.nodeId)}
              onBlockArtist={() => handleBlockArtist(popover.artist)}
              onClose={() => setPopover(null)}
              mobile
            />
          </div>
        ) : (
          <NodePopover
            key={popover.nodeId}
            nodeLabel={popover.label}
            artist={popover.artist}
            trackId={popover.nodeId}
            isSeed={popover.isSeed}
            tags={popover.tags}
            loading={loading}
            onExpand={handleExpand}
            onDelete={() => handleDeleteNode(popover.nodeId)}
            onBlockArtist={() => handleBlockArtist(popover.artist)}
            onClose={() => setPopover(null)}
            x={popover.x}
            y={popover.y}
          />
        )
      )}

      {error && (
        <div className="absolute bottom-24 left-1/2 -translate-x-1/2 z-40 overflow-hidden rounded-xl shadow-[0px_1px_4.1px_0px_rgba(0,0,0,0.25)]">
          <div aria-hidden className="absolute inset-0 backdrop-blur-[4px] bg-white/90 rounded-xl pointer-events-none" />
          <div className="relative px-4 py-2 text-sm text-red-600 flex items-center gap-3">
            {error}
            <button onClick={() => setError(null)} className="text-red-400 hover:text-red-600 transition-colors">
              ×
            </button>
          </div>
        </div>
      )}

      {/* Informational notice (e.g. no recommendations) — neutral, not red, and
          stacked above the error slot so both can coexist. */}
      {notice && (
        <div className="absolute bottom-40 left-1/2 -translate-x-1/2 z-40 max-w-[22rem] overflow-hidden rounded-xl shadow-[0px_1px_4.1px_0px_rgba(0,0,0,0.25)]">
          <div aria-hidden className="absolute inset-0 backdrop-blur-[4px] bg-white/90 rounded-xl pointer-events-none" />
          <div className="relative px-4 py-2 text-sm text-black/70 flex items-start gap-3">
            <span>{notice}</span>
            <button onClick={() => setNotice(null)} className="shrink-0 text-black/30 hover:text-black/60 transition-colors">
              ×
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

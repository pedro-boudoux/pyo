import { useEffect, useState } from "react";
import { Handle, Position, type NodeProps } from "reactflow";

export type SongNodeData = {
  name: string;
  artist: string;
  image: string | null;
  isSeed: boolean;
  similarity?: number;
  listeners?: number;
  // Top tags delivered with the expansion response, so the popover shows them
  // without re-fetching /features (issue #22). Absent on the seed node.
  tags?: string[];
};

export const NODE_SIZE = 190;

type Tone = "light" | "dark";

export function SongNode({ data, selected }: NodeProps<SongNodeData>) {
  const tone = useImageBrightness(data.image);
  const isLight = tone === "light";
  const gradientColor = isLight ? "255,255,255" : "0,0,0";
  const textColorClass = isLight ? "text-neutral-900" : "text-white";
  const subTextColorClass = isLight ? "text-neutral-700" : "text-neutral-300";

  return (
    <div
      style={{ width: NODE_SIZE, height: NODE_SIZE }}
      className={[
        "relative rounded-2xl overflow-hidden transition-shadow",
        selected
          ? "ring-2 ring-white/80 shadow-[0_0_0_4px_rgba(255,255,255,0.25)]"
          : data.isSeed
            ? "ring-2 ring-white/60 shadow-lg"
            : "ring-1 ring-white/30 hover:ring-white/60 shadow-md",
      ].join(" ")}
    >
      <Handle
        type="target"
        position={Position.Top}
        className="!opacity-0 !w-1 !h-1 !min-w-0 !min-h-0 !border-0 !bg-transparent"
      />
      <Handle
        type="source"
        position={Position.Bottom}
        className="!opacity-0 !w-1 !h-1 !min-w-0 !min-h-0 !border-0 !bg-transparent"
      />

      {data.image ? (
        <img
          src={data.image}
          alt=""
          crossOrigin="anonymous"
          referrerPolicy="no-referrer"
          className="absolute inset-0 w-full h-full object-cover select-none"
          draggable={false}
        />
      ) : (
        <div className="absolute inset-0 bg-gradient-to-br from-neutral-700 to-neutral-900" />
      )}

      <div
        className="absolute inset-x-0 bottom-0 h-[65%] pointer-events-none"
        style={{
          background: `linear-gradient(to top, rgba(${gradientColor},0.92) 0%, rgba(${gradientColor},0.7) 40%, rgba(${gradientColor},0) 100%)`,
        }}
      />

      <div className={`absolute inset-x-0 bottom-0 p-3 ${textColorClass}`}>
        <div className="truncate text-sm font-semibold leading-tight">{data.name}</div>
        <div className={`truncate text-[11px] mt-0.5 ${subTextColorClass}`}>{data.artist}</div>
        {data.isSeed ? (
          <div className={`mt-1.5 font-mono text-[9px] uppercase tracking-[0.18em] font-medium ${isLight ? "text-neutral-600" : "text-white/80"}`}>
            Source Song
          </div>
        ) : (
          typeof data.similarity === "number" && (
            <div className={`mt-1 flex justify-between font-mono text-[9px] tabular-nums ${subTextColorClass}`}>
              <span>sim {Math.round(data.similarity * 100)}%</span>
              {typeof data.listeners === "number" && (
                <span>{formatListeners(data.listeners)} listeners</span>
              )}
            </div>
          )
        )}
      </div>
    </div>
  );
}

function formatListeners(n: number): string {
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`;
  if (n >= 1_000) return `${(n / 1_000).toFixed(1)}k`;
  return String(n);
}

function useImageBrightness(src: string | null): Tone {
  const [tone, setTone] = useState<Tone>("dark");

  useEffect(() => {
    if (!src) {
      setTone("dark");
      return;
    }
    let cancelled = false;
    const img = new Image();
    img.crossOrigin = "anonymous";
    img.referrerPolicy = "no-referrer";

    img.onload = () => {
      if (cancelled) return;
      try {
        const size = 32;
        const canvas = document.createElement("canvas");
        canvas.width = size;
        canvas.height = size;
        const ctx = canvas.getContext("2d");
        if (!ctx) return;
        ctx.drawImage(img, 0, 0, size, size);
        const pixels = ctx.getImageData(0, Math.floor(size / 2), size, Math.floor(size / 2)).data;
        let total = 0;
        const count = pixels.length / 4;
        for (let i = 0; i < pixels.length; i += 4) {
          const r = pixels[i];
          const g = pixels[i + 1];
          const b = pixels[i + 2];
          total += (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255;
        }
        const avg = total / count;
        if (!cancelled) setTone(avg > 0.55 ? "light" : "dark");
      } catch {
        // CORS or canvas read failure — keep default dark tone
      }
    };

    img.src = src;
    return () => { cancelled = true; };
  }, [src]);

  return tone;
}

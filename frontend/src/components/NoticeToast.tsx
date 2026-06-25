import { useEffect, useState } from "react";

// Keep in sync with the .toast-out duration in index.css.
const EXIT_MS = 200;

type NoticeToastProps = {
  /** The message to show. `null` plays the exit animation, then unmounts. */
  message: string | null;
  onDismiss: () => void;
};

// A transient toast docked above the search bar (issue #27). It owns its own
// enter/exit animation: when `message` flips to null (auto-dismiss timeout or
// the × button), the text stays mounted long enough to play `toast-out` before
// it actually leaves the DOM, instead of vanishing instantly.
export function NoticeToast({ message, onDismiss }: NoticeToastProps) {
  // The text currently on screen — persists through the exit animation.
  const [shown, setShown] = useState<string | null>(message);
  const [leaving, setLeaving] = useState(false);
  const [prevMessage, setPrevMessage] = useState(message);

  // React to `message` changes during render (React's store-previous-prop
  // pattern) so the synchronous state swap doesn't live in an effect.
  if (message !== prevMessage) {
    setPrevMessage(message);
    if (message) {
      // New (or replacement) message: swap content in and play the entrance.
      setShown(message);
      setLeaving(false);
    } else {
      // Dismissed: play the exit; the effect below schedules the unmount.
      setLeaving(true);
    }
  }

  // Once the exit animation has had time to play, drop the content for good.
  useEffect(() => {
    if (!leaving) return;
    const id = setTimeout(() => setShown(null), EXIT_MS);
    return () => clearTimeout(id);
  }, [leaving]);

  if (!shown) return null;

  return (
    <div className="mb-2 flex justify-center">
      <div
        className={`relative overflow-hidden rounded-xl shadow-[0px_1px_4.1px_0px_rgba(0,0,0,0.25)] max-w-full ${
          leaving ? "toast-out" : "toast-in"
        }`}
      >
        <div aria-hidden className="absolute inset-0 backdrop-blur-[4px] bg-white/90 rounded-xl pointer-events-none" />
        <div className="relative px-4 py-2 text-sm text-black/70 flex items-start gap-3" role="status">
          <span>{shown}</span>
          <button
            onClick={onDismiss}
            className="shrink-0 text-black/30 hover:text-black/60 transition-colors"
            aria-label="Dismiss"
          >
            ×
          </button>
        </div>
      </div>
    </div>
  );
}

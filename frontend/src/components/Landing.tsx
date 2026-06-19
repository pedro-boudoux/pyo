import { useRef } from "react";
import { Hero } from "./Hero";
import { GridBackground } from "./GridBackground";
import type { SongSearchResult } from "../types";

type Props = {
  onPick: (song: SongSearchResult) => void;
  disabled?: boolean;
};

export function Landing({ onPick, disabled }: Props) {
  const aboutRef = useRef<HTMLElement>(null);

  function scrollToAbout() {
    aboutRef.current?.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  return (
    <div className="absolute inset-0 overflow-x-hidden overflow-y-auto">
      {/* Section 1 — hero over the animated grid, one viewport tall */}
      <section className="relative h-full w-full">
        <GridBackground />

        <Hero onPick={onPick} disabled={disabled} />

        <button
          type="button"
          onClick={scrollToAbout}
          aria-label="Scroll down to learn more"
          className="group absolute bottom-8 left-1/2 z-0 -translate-x-1/2 flex flex-col items-center gap-2 text-black/40 hover:text-black/70 transition-colors"
        >
          <span className="text-[11px] uppercase tracking-[0.2em] font-medium">
            About
          </span>
          <span className="scroll-bob">
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" aria-hidden>
              <path
                d="M6 9l6 6 6-6"
                stroke="currentColor"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
              />
            </svg>
          </span>
        </button>
      </section>

      {/* Section 2 — solid white panel for project copy */}
      <section
        ref={aboutRef}
        className="relative min-h-full w-full bg-white flex items-center justify-center px-6 py-20 sm:px-8"
      >
        <div className="grid w-full max-w-5xl items-center gap-12 md:grid-cols-[1.15fr_1fr] md:gap-16">
          {/* Copy */}
          <div className="flex flex-col gap-6">
            <div className="flex flex-col gap-3">
              <span className="text-[11px] font-medium uppercase tracking-[0.2em] text-[#4a90d9]">
                About
              </span>
              <h2 className="font-display text-4xl leading-tight text-[#1a1a1a] sm:text-5xl">
                Thank you for using pyo!
              </h2>
            </div>

            <div className="flex flex-col gap-4 text-[15px] leading-relaxed text-black/70 sm:text-base">
              <p>
                pyo (for <span className="italic text-black/80">putting you on</span>) is a music discovery tool I built out of frustration with music algorithms. You know that moment when you're pretty much in flow state on a serious streak of good music and then the algorithm decides its Drake time? Yeah, that's why I made pyo 😭🙏🏼.
              </p>
              <p>
                Using pyo is quite simple: you drop a song you like, and pyo builds a graph connecting it to similar music. Like one of the recommendations? Extend the graph from there and keep going.
              </p>
              <p>
                If pyo puts you on to something good, consider giving it a star on&nbsp;
                <a
                  href="https://github.com/pedro-boudoux/pyo"
                  target="_blank"
                  rel="noreferrer"
                  className="font-medium text-[#4a90d9] underline-offset-2 hover:underline"
                >
                   GitHub ⭐️
                </a>
                , it would be greatly appreciated. You can also find a deeper breakdown of how the algorithm works there.
              </p>
            </div>

            {/* Contact */}
            <div className="flex flex-col gap-3 pt-2">
              <p className="text-[15px] leading-relaxed text-black/70 sm:text-base">
                Questions or just want to say hi?
              </p>
              <div className="flex flex-wrap gap-2.5">
                <a
                  href="mailto:me@pedroboudoux.com"
                  className="flex items-center gap-2 rounded-full border border-black/10 bg-black/[0.02] px-4 py-2 text-sm font-medium text-black/70 transition-colors hover:border-[#4a90d9]/40 hover:bg-[#4a90d9]/[0.06] hover:text-[#4a90d9]"
                >
                  <svg viewBox="0 0 24 24" className="h-4 w-4" fill="none" stroke="currentColor" strokeWidth={1.8} aria-hidden>
                    <rect x="3" y="5" width="18" height="14" rx="2" />
                    <path d="m3 7 9 6 9-6" strokeLinecap="round" strokeLinejoin="round" />
                  </svg>
                  Email
                </a>
                <a
                  href="https://x.com/pbdoux"
                  target="_blank"
                  rel="noreferrer"
                  className="flex items-center gap-2 rounded-full border border-black/10 bg-black/[0.02] px-4 py-2 text-sm font-medium text-black/70 transition-colors hover:border-[#4a90d9]/40 hover:bg-[#4a90d9]/[0.06] hover:text-[#4a90d9]"
                >
                  <svg viewBox="0 0 24 24" className="h-4 w-4" fill="currentColor" aria-hidden>
                    <path d="M18.244 2.25h3.308l-7.227 8.26 8.502 11.24h-6.66l-5.214-6.817L4.99 21.75H1.68l7.73-8.835L1.254 2.25H8.08l4.713 6.231zm-1.161 17.52h1.833L7.084 4.126H5.117z" />
                  </svg>
                  Twitter / X
                </a>
                <a
                  href="https://www.linkedin.com/in/pedroboudoux/"
                  target="_blank"
                  rel="noreferrer"
                  className="flex items-center gap-2 rounded-full border border-black/10 bg-black/[0.02] px-4 py-2 text-sm font-medium text-black/70 transition-colors hover:border-[#4a90d9]/40 hover:bg-[#4a90d9]/[0.06] hover:text-[#4a90d9]"
                >
                  <svg viewBox="0 0 24 24" className="h-4 w-4" fill="currentColor" aria-hidden>
                    <path d="M20.447 20.452h-3.554v-5.569c0-1.328-.027-3.037-1.852-3.037-1.853 0-2.136 1.445-2.136 2.939v5.667H9.351V9h3.414v1.561h.046c.477-.9 1.637-1.85 3.37-1.85 3.601 0 4.267 2.37 4.267 5.455v6.286zM5.337 7.433a2.062 2.062 0 0 1-2.063-2.065 2.064 2.064 0 1 1 2.063 2.065zm1.782 13.019H3.555V9h3.564v11.452zM22.225 0H1.771C.792 0 0 .774 0 1.729v20.542C0 23.227.792 24 1.771 24h20.451C23.2 24 24 23.227 24 22.271V1.729C24 .774 23.2 0 22.225 0z" />
                  </svg>
                  LinkedIn
                </a>
              </div>
            </div>
          </div>

          {/* Images */}
          <div className="flex flex-col items-center gap-6">
            <img
              className="h-auto w-full max-w-[360px]"
              src={`${import.meta.env.BASE_URL}images/tiffany-day.png`}
              alt="Pyo graph example."
            />
            <img
              className="h-auto w-full max-w-[360px]"
              src={`${import.meta.env.BASE_URL}images/charlie-brown-jr.png`}
              alt="Pyo graph example."
            />
          </div>
        </div>
      </section>
    </div>
  );
}

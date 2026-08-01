import { afterEach, describe, expect, it, vi } from "vitest";
import { rejectTrack } from "./api";

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("rejectTrack", () => {
  it("sends the explicit parent in the feedback contract", async () => {
    const fetchMock = vi.fn(
      async (input: RequestInfo | URL, init?: RequestInit) => {
        void input;
        void init;
        return new Response(
          JSON.stringify({ success: true, message: "saved" }),
          { status: 200, headers: { "Content-Type": "application/json" } },
        );
      },
    );
    vi.stubGlobal("fetch", fetchMock);

    await rejectTrack("parent-a", "child");

    const [, init] = fetchMock.mock.calls[0];
    expect(init?.method).toBe("POST");
    expect(JSON.parse(String(init?.body))).toEqual({
      source_track_id: "parent-a",
      track_id: "child",
      action: "reject",
    });
  });
});

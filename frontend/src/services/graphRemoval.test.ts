import { describe, expect, it, vi } from "vitest";
import {
  incomingParentIds,
  feedbackParentIds,
  persistIncomingRejections,
  removeAfterPersistingRejections,
} from "./graphRemoval";

const edges = [
  { source: "parent-a", target: "child" },
  { source: "parent-b", target: "child" },
  { source: "parent-a", target: "child" },
  { source: "child", target: "grandchild" },
];

describe("parent-scoped graph removal", () => {
  it("finds every unique visible incoming parent", () => {
    expect(incomingParentIds("child", edges)).toEqual(["parent-a", "parent-b"]);
  });

  it("submits one rejection for each incoming parent", async () => {
    const submit = vi.fn(async () => undefined);

    await persistIncomingRejections("child", edges, submit);

    expect(submit.mock.calls).toEqual([
      ["parent-a", "child"],
      ["parent-b", "child"],
    ]);
  });

  it("creates no feedback for seeds or automatic removal reasons", () => {
    expect(feedbackParentIds("child", edges, {
      isSeed: true,
      reason: "deliberate",
    })).toEqual([]);

    for (const reason of ["orphan-prune", "artist-block", "restart"] as const) {
      expect(feedbackParentIds("child", edges, {
        isSeed: false,
        reason,
      })).toEqual([]);
    }
  });

  it("keeps the node on failure and safely retries every parent", async () => {
    const attempts = new Map<string, number>();
    const submit = vi.fn(async (source: string) => {
      attempts.set(source, (attempts.get(source) ?? 0) + 1);
      if (source === "parent-b" && attempts.get(source) === 1) {
        throw new Error("temporary failure");
      }
    });
    const remove = vi.fn();

    await expect(
      removeAfterPersistingRejections("child", edges, submit, remove),
    ).rejects.toThrow("temporary failure");
    expect(remove).not.toHaveBeenCalled();

    await removeAfterPersistingRejections("child", edges, submit, remove);
    expect(remove).toHaveBeenCalledOnce();
    expect(attempts).toEqual(new Map([
      ["parent-a", 2],
      ["parent-b", 2],
    ]));
  });
});

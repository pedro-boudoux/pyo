type DirectedEdge = {
  source?: string | number | null;
  target?: string | number | null;
};

export type SubmitRejection = (
  sourceTrackId: string,
  rejectedTrackId: string,
) => Promise<unknown>;

export type RemovalReason =
  | "deliberate"
  | "orphan-prune"
  | "artist-block"
  | "restart";

type RemovalContext = {
  isSeed: boolean;
  reason: RemovalReason;
};

export function incomingParentIds(
  rejectedTrackId: string,
  edges: DirectedEdge[],
): string[] {
  return [
    ...new Set(
      edges
        .filter((edge) => String(edge.target) === rejectedTrackId)
        .map((edge) => String(edge.source))
        .filter((source) => source && source !== "null" && source !== "undefined"),
    ),
  ];
}

export function feedbackParentIds(
  rejectedTrackId: string,
  edges: DirectedEdge[],
  context: RemovalContext,
): string[] {
  if (context.isSeed || context.reason !== "deliberate") return [];
  return incomingParentIds(rejectedTrackId, edges);
}

/** Persist every visible parent-scoped rejection before local graph mutation.
 *
 * Every request is allowed to settle even if one parent fails. The caller keeps
 * the node on any failure; a retry may resubmit successful parents, which the
 * backend accepts idempotently.
 */
export async function persistIncomingRejections(
  rejectedTrackId: string,
  edges: DirectedEdge[],
  submit: SubmitRejection,
  context: RemovalContext = { isSeed: false, reason: "deliberate" },
): Promise<string[]> {
  const parents = feedbackParentIds(rejectedTrackId, edges, context);
  const results = await Promise.allSettled(
    parents.map((parent) => submit(parent, rejectedTrackId)),
  );
  const failure = results.find(
    (result): result is PromiseRejectedResult => result.status === "rejected",
  );
  if (failure) throw failure.reason;
  return parents;
}

export async function removeAfterPersistingRejections(
  rejectedTrackId: string,
  edges: DirectedEdge[],
  submit: SubmitRejection,
  remove: () => void,
  context: RemovalContext = { isSeed: false, reason: "deliberate" },
): Promise<string[]> {
  const parents = await persistIncomingRejections(
    rejectedTrackId,
    edges,
    submit,
    context,
  );
  remove();
  return parents;
}

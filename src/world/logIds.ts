/**
 * Known log block runtime IDs.
 *
 * The chunk decoder only gives us numeric runtime IDs, no names, so the tree
 * detector used to be purely geometric and false-positived on stone pillars and
 * spawn structures. Instead we positively identify logs by runtime id.
 *
 * 12971 = confirmed oak log on this BDS world (captured from a real break).
 * Runtime IDs differ across BDS versions, so this set is meant to grow at
 * runtime via addLogId once we wire up auto-learning from break events.
 */
const SEED_LOG_IDS: readonly number[] = [12971];
const LOG_IDS = new Set<number>(SEED_LOG_IDS);

/** True if the given runtime id is a known log. */
export function isLogId(id: number): boolean {
  return LOG_IDS.has(id);
}

/** Register a runtime id as a known log (auto-learned by findNearbyTree). */
export function addLogId(id: number): void {
  LOG_IDS.add(id);
}

/** Drop all auto-learned ids, restoring the seed set. The learned set is
 * process-global (we remember a species once we've seen it), so tests that
 * exercise auto-learning must reset it to stay isolated from each other. */
export function resetLearnedLogIds(): void {
  LOG_IDS.clear();
  for (const id of SEED_LOG_IDS) LOG_IDS.add(id);
}

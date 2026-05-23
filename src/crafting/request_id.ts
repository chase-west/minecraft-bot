/**
 * Bedrock item_stack_request requires unique, odd, negative request_ids
 * (zigzag32-encoded). Servers reject reused IDs or positive ones.
 * We allocate -1, -3, -5, ... per session.
 */
let next = -1;

export function nextRequestId(): number {
  const id = next;
  next -= 2;
  if (next > -3) next = -3; // guard against wrap (won't happen in practice for ~2B crafts)
  return id;
}

export function resetRequestIds(): void {
  next = -1;
}

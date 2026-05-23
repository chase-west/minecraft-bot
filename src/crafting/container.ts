import type { BedrockClient } from "../connection/client.js";
import type { World } from "../world/world.js";
import type { Vec3 } from "../utils/vec3.js";
import { safeQueue } from "../connection/version.js";
import { makeLogger } from "../utils/logger.js";

const log = makeLogger("container");

let openWindowId: number | null = null;

export function attachContainerEvents(client: BedrockClient): void {
  client.on("container_open", (pkt: any) => {
    openWindowId = pkt.window_id ?? null;
    log.info(`container opened: window=${openWindowId} type=${pkt.window_type}`);
  });
  client.on("container_close", (pkt: any) => {
    log.info(`container closed: window=${pkt.window_id}`);
    openWindowId = null;
  });
}

/** Right-click a crafting table block to open its 3x3 grid. */
export async function openCraftingTable(client: BedrockClient, world: World, tablePos: Vec3): Promise<{ opened: boolean; reason?: string }> {
  // inventory_transaction with item_use action=click_block on the table.
  const heldSlot = world.selectedHotbarSlot;
  safeQueue(client, "inventory_transaction", {
    transaction: {
      legacy: { legacy_request_id: 0 },
      transaction_type: "item_use",
      actions: [],
      transaction_data: {
        action_type: 0, // click_block
        block_position: { x: tablePos.x, y: tablePos.y, z: tablePos.z },
        face: 1,
        hotbar_slot: heldSlot,
        held_item: {
          network_id: world.inventory.get(heldSlot)?.networkId ?? 0,
          count: world.inventory.get(heldSlot)?.count ?? 0,
          metadata: 0,
          has_stack_id: 0,
          block_runtime_id: 0,
          extra: { has_nbt: 0, can_place_on: [], can_destroy: [] },
        },
        player_pos: world.self.position,
        click_pos: { x: tablePos.x + 0.5, y: tablePos.y + 0.5, z: tablePos.z + 0.5 },
        block_runtime_id: 0,
      },
    },
  }, "open_table");

  // Wait for container_open.
  const deadline = Date.now() + 2000;
  while (Date.now() < deadline) {
    if (openWindowId !== null) return { opened: true };
    await new Promise((r) => setTimeout(r, 50));
  }
  return { opened: false, reason: "no_container_open" };
}

export async function closeContainer(client: BedrockClient): Promise<void> {
  if (openWindowId === null) return;
  safeQueue(client, "container_close", { window_id: openWindowId, server: false }, "close");
  openWindowId = null;
  await new Promise((r) => setTimeout(r, 200));
}

export function isContainerOpen(): boolean {
  return openWindowId !== null;
}

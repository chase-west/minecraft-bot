/**
 * Minimal bedrock subchunk binary decoder.
 *
 * Wire format (per bedrock-protocol docs + prismarine-chunk reference):
 *   - version: u8 (1, 8, or 9)
 *   - storage_count: u8  (1 if version == 1)
 *   - if version == 9: y_index: i8
 *   - per storage layer:
 *       - palette_header: u8 = (bits_per_block << 1) | network_flag
 *       - if bits_per_block == 0: single varint runtime_id (4096 blocks all identical)
 *       - else:
 *           - 4096 indices packed into ceil(4096 / floor(32/bpb)) u32 LE words
 *           - palette_length: zigzag-varint
 *           - palette_length × zigzag-varint runtime_ids
 *
 * Index ordering within a 16×16×16 subchunk is XZY:
 *   index = (x << 8) | (z << 4) | y
 *
 * We decode ONLY the first storage layer (block layer; layer 1 if present is liquid overlay)
 * and return raw runtime_id integers. Naming is resolved separately (see name_inference.ts).
 */

export interface SubChunkBlock {
  x: number; // 0..15 within subchunk
  y: number; // 0..15 within subchunk
  z: number; // 0..15 within subchunk
  runtimeId: number;
}

export interface SubChunkDecoded {
  yIndex: number;             // signed y index of this subchunk (16-block units)
  blocks: SubChunkBlock[];    // only non-zero (non-air) blocks
  palette: number[];          // unique runtime IDs that appeared in this subchunk
}

class Cursor {
  constructor(public buf: Buffer, public off = 0) {}
  u8(): number { return this.buf.readUInt8(this.off++); }
  i8(): number { return this.buf.readInt8(this.off++); }
  u32le(): number { const v = this.buf.readUInt32LE(this.off); this.off += 4; return v; }
  varint(): number {
    let result = 0; let shift = 0; let b: number;
    do {
      b = this.buf.readUInt8(this.off++);
      result |= (b & 0x7f) << shift;
      shift += 7;
      if (shift > 35) throw new Error("varint too long");
    } while (b & 0x80);
    return result >>> 0;
  }
  zigzag(): number {
    const v = this.varint();
    return (v >>> 1) ^ -(v & 1);
  }
}

/** Decode a single subchunk payload (as received in a `subchunk` packet entry). */
export function decodeSubChunk(payload: Buffer, defaultYIndex = 0): SubChunkDecoded {
  if (payload.length === 0) {
    return { yIndex: defaultYIndex, blocks: [], palette: [] };
  }
  const c = new Cursor(payload);
  const version = c.u8();
  if (version !== 1 && version !== 8 && version !== 9) {
    throw new Error(`unsupported subchunk version ${version}`);
  }
  const storageCount = version === 1 ? 1 : c.u8();
  const yIndex = version === 9 ? c.i8() : defaultYIndex;

  let firstLayerRuntimeIds: number[] | null = null;

  for (let layer = 0; layer < storageCount; layer++) {
    const header = c.u8();
    const bitsPerBlock = header >>> 1;
    // Bits 0 of header = network flag; we expect 1 for network format.

    if (bitsPerBlock === 0) {
      // Uniform — every block is the same. Single varint.
      const idRaw = c.varint();
      const runtimeId = idRaw >>> 1; // zigzag decode of u32 varint
      if (layer === 0) {
        firstLayerRuntimeIds = new Array<number>(4096).fill(runtimeId);
      }
      continue;
    }

    const blocksPerWord = Math.floor(32 / bitsPerBlock);
    const wordsCount = Math.ceil(4096 / blocksPerWord);
    const mask = (1 << bitsPerBlock) - 1;
    const indices = new Array<number>(4096);

    // Read all words first into a local buffer view.
    const wordsStart = c.off;
    for (let i = 0; i < 4096; i++) {
      const word = c.buf.readUInt32LE(wordsStart + Math.floor(i / blocksPerWord) * 4);
      indices[i] = (word >>> ((i % blocksPerWord) * bitsPerBlock)) & mask;
    }
    c.off = wordsStart + wordsCount * 4;

    // Read palette: length then entries (each zigzag varint).
    const palLen = c.zigzag();
    if (palLen <= 0 || palLen > 65536) {
      throw new Error(`bogus palette length ${palLen}`);
    }
    const palette = new Array<number>(palLen);
    for (let p = 0; p < palLen; p++) {
      palette[p] = c.zigzag();
    }

    if (layer === 0) {
      const out = new Array<number>(4096);
      for (let i = 0; i < 4096; i++) {
        const pi = indices[i]!;
        out[i] = palette[pi] ?? 0;
      }
      firstLayerRuntimeIds = out;
    }
  }

  if (!firstLayerRuntimeIds) {
    return { yIndex, blocks: [], palette: [] };
  }

  const blocks: SubChunkBlock[] = [];
  const seenIds = new Set<number>();
  for (let x = 0; x < 16; x++) {
    for (let z = 0; z < 16; z++) {
      for (let y = 0; y < 16; y++) {
        const idx = (x << 8) | (z << 4) | y;
        const rid = firstLayerRuntimeIds[idx]!;
        if (rid === 0) continue; // air
        seenIds.add(rid);
        blocks.push({ x, y, z, runtimeId: rid });
      }
    }
  }
  return { yIndex, blocks, palette: [...seenIds] };
}

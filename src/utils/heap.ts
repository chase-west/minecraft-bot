export class MinHeap<T> {
  private heap: T[] = [];
  constructor(private readonly compare: (a: T, b: T) => number) {}

  get size(): number {
    return this.heap.length;
  }

  push(item: T): void {
    this.heap.push(item);
    this.siftUp(this.heap.length - 1);
  }

  pop(): T | undefined {
    const n = this.heap.length;
    if (n === 0) return undefined;
    const top = this.heap[0];
    const last = this.heap.pop()!;
    if (n > 1) {
      this.heap[0] = last;
      this.siftDown(0);
    }
    return top;
  }

  peek(): T | undefined {
    return this.heap[0];
  }

  private siftUp(i: number): void {
    while (i > 0) {
      const parent = (i - 1) >> 1;
      if (this.compare(this.heap[i]!, this.heap[parent]!) < 0) {
        [this.heap[i], this.heap[parent]] = [this.heap[parent]!, this.heap[i]!];
        i = parent;
      } else break;
    }
  }

  private siftDown(i: number): void {
    const n = this.heap.length;
    while (true) {
      const l = 2 * i + 1;
      const r = 2 * i + 2;
      let smallest = i;
      if (l < n && this.compare(this.heap[l]!, this.heap[smallest]!) < 0) smallest = l;
      if (r < n && this.compare(this.heap[r]!, this.heap[smallest]!) < 0) smallest = r;
      if (smallest !== i) {
        [this.heap[i], this.heap[smallest]] = [this.heap[smallest]!, this.heap[i]!];
        i = smallest;
      } else break;
    }
  }
}

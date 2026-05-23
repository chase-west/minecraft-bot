export interface Vec3 {
  x: number;
  y: number;
  z: number;
}

export const v3 = (x: number, y: number, z: number): Vec3 => ({ x, y, z });

export const v3add = (a: Vec3, b: Vec3): Vec3 => ({ x: a.x + b.x, y: a.y + b.y, z: a.z + b.z });
export const v3sub = (a: Vec3, b: Vec3): Vec3 => ({ x: a.x - b.x, y: a.y - b.y, z: a.z - b.z });
export const v3scale = (a: Vec3, s: number): Vec3 => ({ x: a.x * s, y: a.y * s, z: a.z * s });

export const v3dist = (a: Vec3, b: Vec3): number => {
  const dx = a.x - b.x, dy = a.y - b.y, dz = a.z - b.z;
  return Math.sqrt(dx * dx + dy * dy + dz * dz);
};

export const v3distXZ = (a: Vec3, b: Vec3): number => {
  const dx = a.x - b.x, dz = a.z - b.z;
  return Math.sqrt(dx * dx + dz * dz);
};

export const v3floor = (a: Vec3): Vec3 => ({
  x: Math.floor(a.x),
  y: Math.floor(a.y),
  z: Math.floor(a.z),
});

export const v3key = (a: Vec3): string => `${a.x | 0},${a.y | 0},${a.z | 0}`;

export const v3equal = (a: Vec3, b: Vec3): boolean =>
  a.x === b.x && a.y === b.y && a.z === b.z;

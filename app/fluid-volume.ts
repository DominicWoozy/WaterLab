/** A world-space scalar field reconstructed from the physical particles.
 * It is independent of the camera and does not use visible sphere surfaces.
 */
export const VOLUME_SIZE = [96, 128, 72] as const;
export const VOLUME_MIN = [-2.08, -1.12, -1.56] as const;
export const VOLUME_MAX = [2.08, 4.08, 1.56] as const;
export const SURFACE_DENSITY = 1.15;
export const ABSORPTION = [1.25, 0.2, 0.065] as const;
const RADIUS = 0.19;
const STEP = VOLUME_SIZE.map(
  (n, i) => (VOLUME_MAX[i] - VOLUME_MIN[i]) / (n - 1),
);
export class FluidVolume {
  data = new Float32Array(VOLUME_SIZE[0] * VOLUME_SIZE[1] * VOLUME_SIZE[2]);
  private xSquared = new Float32Array(VOLUME_SIZE[0]);
  top = 0;
  rebuild(positions: Float32Array, count: number, densities?: Float32Array) {
    this.data.fill(0);
    const [nx, ny, nz] = VOLUME_SIZE,
      [sx, sy, sz] = STEP;
    const h2 = RADIUS * RADIUS;
    this.top = VOLUME_MIN[1];
    for (let i = 0; i < count; i++) {
      const x = positions[i * 3],
        y = positions[i * 3 + 1],
        z = positions[i * 3 + 2];
      this.top = Math.max(this.top, Math.min(VOLUME_MAX[1], y + RADIUS));
      const x0 = Math.max(0, Math.ceil((x - RADIUS - VOLUME_MIN[0]) / sx));
      const x1 = Math.min(
        nx - 1,
        Math.floor((x + RADIUS - VOLUME_MIN[0]) / sx),
      );
      const y0 = Math.max(0, Math.ceil((y - RADIUS - VOLUME_MIN[1]) / sy));
      const y1 = Math.min(
        ny - 1,
        Math.floor((y + RADIUS - VOLUME_MIN[1]) / sy),
      );
      const z0 = Math.max(0, Math.ceil((z - RADIUS - VOLUME_MIN[2]) / sz));
      const z1 = Math.min(
        nz - 1,
        Math.floor((z + RADIUS - VOLUME_MIN[2]) / sz),
      );
      for (let ix = x0; ix <= x1; ix++) {
        const dx = VOLUME_MIN[0] + ix * sx - x;
        this.xSquared[ix] = (dx * dx) / h2;
      }
      // Only detached spray needs extra support to survive the grid resolution.
      const weight = densities ? 1 + Math.max(0, 1 - densities[i]) * 0.8 : 1;
      for (let iz = z0; iz <= z1; iz++) {
        const dz = VOLUME_MIN[2] + iz * sz - z;
        for (let iy = y0; iy <= y1; iy++) {
          const dy = VOLUME_MIN[1] + iy * sy - y,
            yz2 = (dy * dy + dz * dz) / h2;
          if (yz2 >= 1) continue;
          let index = nx * (iy + ny * iz) + x0;
          for (let ix = x0; ix <= x1; ix++, index++) {
            const q = 1 - yz2 - this.xSquared[ix];
            if (q > 0) this.data[index] += q * q * q * weight;
          }
        }
      }
    }
  }
  // Matches the GPU's trilinear field sampling, also used by numerical checks.
  sample(x: number, y: number, z: number) {
    const p = [x, y, z].map((v, i) => (v - VOLUME_MIN[i]) / STEP[i]);
    if (p.some((v, i) => v < 0 || v > VOLUME_SIZE[i] - 1)) return 0;
    const cell = p.map((v, i) => Math.min(VOLUME_SIZE[i] - 2, Math.floor(v))),
      f = p.map((v, i) => v - cell[i]);
    let density = 0;
    for (let iz = 0; iz < 2; iz++)
      for (let iy = 0; iy < 2; iy++)
        for (let ix = 0; ix < 2; ix++) {
          const weight =
            (ix ? f[0] : 1 - f[0]) *
            (iy ? f[1] : 1 - f[1]) *
            (iz ? f[2] : 1 - f[2]);
          density +=
            this.data[
              cell[0] +
                ix +
                VOLUME_SIZE[0] *
                  (cell[1] + iy + VOLUME_SIZE[1] * (cell[2] + iz))
            ] * weight;
        }
    return density;
  }
  thickness(
    origin: number[],
    direction: number[],
    distance: number,
    step = 0.015,
  ) {
    let length = 0;
    for (let t = step / 2; t < distance; t += step) {
      const d = this.sample(
        origin[0] + direction[0] * t,
        origin[1] + direction[1] * t,
        origin[2] + direction[2] * t,
      );
      const q = Math.max(0, Math.min(1, (d - (SURFACE_DENSITY - 0.18)) / 0.36));
      length += q * q * (3 - 2 * q) * step;
    }
    return length;
  }
}
export const transmittance = (thickness: number) =>
  ABSORPTION.map((coefficient) =>
    Math.exp(-coefficient * Math.max(0, thickness)),
  );

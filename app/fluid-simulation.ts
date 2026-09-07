/** Particle fluid with spatial counting sort and position-based density projection.
 * Compression constraints; no elastic springs or tensile attraction.
 * Fixed timesteps and pairwise symmetric corrections preserve stable motion.
 */
export const CAPACITY = 12000;
export const DEFAULT_COUNT = 10000;
export const PARTICLE_SPACING = 0.09;
export const FLOOR = -0.95;
export const HALF_X = 1.78;
export const HALF_Z = 1.28;
export const KERNEL_RADIUS = 0.17;
const H = KERNEL_RADIUS;
const H2 = H * H;
const NX = 26,
  NY = 32,
  NZ = 21;
const clamp = (x: number, lo: number, hi: number) =>
  Math.max(lo, Math.min(hi, x));
export type FluidForces = {
  gravity: number;
  viscosity: number;
  agitation: number;
};
export class ParticleFluid {
  readonly positions = new Float32Array(CAPACITY * 3);
  readonly velocities = new Float32Array(CAPACITY * 3);
  readonly densities = new Float32Array(CAPACITY);
  private previous = new Float32Array(CAPACITY * 3);
  private corrections = new Float32Array(CAPACITY * 3);
  private pressure = new Float32Array(CAPACITY);
  private gradient = new Float32Array(CAPACITY * 3);
  private gradientSquared = new Float32Array(CAPACITY);
  private cellCounts = new Int32Array(NX * NY * NZ);
  private cellStarts = new Int32Array(NX * NY * NZ + 1);
  private cellCursor = new Int32Array(NX * NY * NZ);
  private particleCells = new Int32Array(CAPACITY);
  private sorted = new Uint16Array(CAPACITY);
  private occupied = new Int32Array(NX * NY * NZ);
  private neighborOffsets: number[] = [];
  private occupiedCount = 0;
  private pairI = new Uint16Array(CAPACITY * 90);
  private pairJ = new Uint16Array(CAPACITY * 90);
  private pairQ = new Float32Array(CAPACITY * 90);
  private pairInvR = new Float32Array(CAPACITY * 90);
  private pairCount = 0;
  private seed = 71429;
  count = 0;
  time = 0;
  private shakeUntil = 0;

  constructor(count = DEFAULT_COUNT) {
    for (let z = -1; z <= 1; z++)
      for (let y = -1; y <= 1; y++)
        for (let x = -1; x <= 1; x++) {
          const offset = x + NX * (y + NY * z);
          if (offset > 0) this.neighborOffsets.push(offset);
        }
    this.reset(count);
  }
  private random() {
    this.seed = (Math.imul(this.seed, 1664525) + 1013904223) >>> 0;
    return this.seed / 4294967296;
  }
  reset(count = DEFAULT_COUNT) {
    this.count = 0;
    this.time = 0;
    this.seed = 71429;
    this.shakeUntil = 0;
    this.velocities.fill(0);
    // Begin with a level basin, not an elastic column or a moving solid block.
    this.densities.fill(3.6);
    const amount = clamp(Math.round(count), 0, CAPACITY);
    const columns = 50,
      rows = 40;
    for (let i = 0; i < amount; i++) {
      const column = i % (columns * rows),
        layer = Math.floor(i / (columns * rows));
      const x = column % columns,
        z = Math.floor(column / columns);
      this.add(
        (x - (columns - 1) / 2) * (3.4 / 49),
        FLOOR + 0.045 + layer * PARTICLE_SPACING,
        (z - (rows - 1) / 2) * (2.4 / 39),
        0,
        0,
        0,
      );
    }
  }
  private add(
    x: number,
    y: number,
    z: number,
    vx: number,
    vy: number,
    vz: number,
  ) {
    if (this.count >= CAPACITY) return;
    const particle = this.count++;
    this.densities[particle] = 3.6;
    const i = particle * 3;
    this.positions.set(
      [
        clamp(x, -HALF_X, HALF_X),
        clamp(y, FLOOR + 0.04, 3.8),
        clamp(z, -HALF_Z, HALF_Z),
      ],
      i,
    );
    this.velocities.set([vx, vy, vz], i);
  }
  pour(x = -0.75, z = 0, amount = 12) {
    const oldCount = this.count;
    for (let i = 0; i < amount && this.count < CAPACITY; i++) {
      this.add(
        x + (this.random() - 0.5) * 0.29,
        1.8 + this.random() * 0.25,
        z + (this.random() - 0.5) * 0.29,
        0.5,
        -1.8,
        0.1,
      );
    }
    return this.count - oldCount;
  }
  drain(amount = 500) {
    this.count = Math.max(0, this.count - Math.max(0, Math.round(amount)));
  }
  shake() {
    this.shakeUntil = this.time + 1.6;
  }
  stir(
    x: number,
    y: number,
    z: number,
    dx: number,
    dz: number,
    strength: number,
    dt: number,
  ) {
    const p = this.positions,
      v = this.velocities;
    for (let i = 0; i < this.count * 3; i += 3) {
      const px = p[i] - x,
        py = (p[i + 1] - y) * 0.45,
        pz = p[i + 2] - z;
      const r2 = px * px + py * py + pz * pz;
      if (r2 > 1.1) continue;
      const force = Math.exp(-r2 * 3.5) * strength * dt;
      // Tangential force keeps a held brush moving, while pointer motion pushes the fluid.
      v[i] += (clamp(dx, -10, 10) * 35 - pz * 24) * force;
      v[i + 1] += 15 * force;
      v[i + 2] += (clamp(dz, -10, 10) * 35 + px * 24) * force;
    }
  }
  splash(x = 0, z = 0, strength = 1) {
    for (let i = 0; i < this.count * 3; i += 3) {
      const dx = this.positions[i] - x,
        dz = this.positions[i + 2] - z;
      const r2 = dx * dx + dz * dz;
      if (r2 > 1.2) continue;
      const w = Math.exp(-r2 * 4) * strength;
      this.velocities[i] += dx * 3.5 * w;
      this.velocities[i + 1] += 4.6 * w;
      this.velocities[i + 2] += dz * 3.5 * w;
    }
  }
  private addPair(i: number, j: number) {
    const p = this.positions,
      a = i * 3,
      b = j * 3;
    const dx = p[b] - p[a],
      dy = p[b + 1] - p[a + 1],
      dz = p[b + 2] - p[a + 2];
    const r2 = dx * dx + dy * dy + dz * dz;
    if (r2 >= H2) return;
    if (this.pairCount === this.pairI.length) this.growPairs();
    const pair = this.pairCount++,
      r = Math.max(0.00001, Math.sqrt(r2));
    this.pairI[pair] = i;
    this.pairJ[pair] = j;
    this.pairQ[pair] = 1 - r / H;
    this.pairInvR[pair] = 1 / r;
  }
  private growPairs() {
    // Never silently drop neighbor interactions in locally dense splashes.
    const size = this.pairI.length * 2;
    const pi = new Uint16Array(size),
      pj = new Uint16Array(size),
      q = new Float32Array(size),
      inv = new Float32Array(size);
    pi.set(this.pairI);
    pj.set(this.pairJ);
    q.set(this.pairQ);
    inv.set(this.pairInvR);
    this.pairI = pi;
    this.pairJ = pj;
    this.pairQ = q;
    this.pairInvR = inv;
  }
  private buildPairs() {
    const p = this.positions,
      counts = this.cellCounts,
      starts = this.cellStarts;
    counts.fill(0);
    this.occupiedCount = 0;
    for (let i = 0; i < this.count; i++) {
      const k = i * 3;
      // One-cell padding makes the 13 forward neighbor offsets safe.
      const x = clamp(Math.floor((p[k] + 2.1) / H), 1, NX - 2);
      const y = clamp(Math.floor((p[k + 1] + 1.1) / H), 1, NY - 2);
      const z = clamp(Math.floor((p[k + 2] + 1.6) / H), 1, NZ - 2);
      const cell = x + NX * (y + NY * z);
      this.particleCells[i] = cell;
      if (counts[cell]++ === 0) this.occupied[this.occupiedCount++] = cell;
    }
    starts[0] = 0;
    for (let c = 0; c < counts.length; c++) {
      starts[c + 1] = starts[c] + counts[c];
      this.cellCursor[c] = starts[c];
    }
    for (let i = 0; i < this.count; i++)
      this.sorted[this.cellCursor[this.particleCells[i]]++] = i;
    this.pairCount = 0;
    for (let c = 0; c < this.occupiedCount; c++) {
      const cell = this.occupied[c],
        begin = starts[cell],
        end = starts[cell + 1];
      for (let a = begin; a < end; a++)
        for (let b = a + 1; b < end; b++)
          this.addPair(this.sorted[a], this.sorted[b]);
      for (let n = 0; n < this.neighborOffsets.length; n++) {
        const other = cell + this.neighborOffsets[n];
        if (!counts[other]) continue;
        for (let a = begin; a < end; a++)
          for (let b = starts[other]; b < starts[other + 1]; b++)
            this.addPair(this.sorted[a], this.sorted[b]);
      }
    }
  }
  step(dt: number, forces: FluidForces) {
    // The caller accumulates wall-clock time; the solver never takes a large step.
    if (!(dt > 0 && dt <= 1 / 60))
      throw new Error('Fluid step must be between 0 and 1/60 seconds');
    const p = this.positions,
      v = this.velocities,
      prev = this.previous,
      corr = this.corrections;
    this.time += dt;
    const shake =
      this.time < this.shakeUntil ? 22 * Math.sin(this.time * 17) : 0;
    const windX = Math.sin(this.time * 2.1) * forces.agitation * 3 + shake;
    const windZ = Math.cos(this.time * 1.7) * forces.agitation * 1.7;
    for (let k = 0; k < this.count * 3; k += 3) {
      prev[k] = p[k];
      prev[k + 1] = p[k + 1];
      prev[k + 2] = p[k + 2];
      v[k] += windX * dt;
      v[k + 1] -= forces.gravity * dt;
      v[k + 2] += windZ * dt;
      const speed = Math.sqrt(
        v[k] * v[k] + v[k + 1] * v[k + 1] + v[k + 2] * v[k + 2],
      );
      const limit = speed > 12 ? 12 / speed : 1;
      p[k] += v[k] * limit * dt;
      p[k + 1] += v[k + 1] * limit * dt;
      p[k + 2] += v[k + 2] * limit * dt;
    }
    this.buildPairs();
    // Position-based density projection avoids a stiffness coefficient exploding at high resolution.
    for (let iteration = 0; iteration < 2; iteration++) {
      this.densities.fill(0);
      this.gradient.fill(0);
      this.gradientSquared.fill(0);
      corr.fill(0);
      for (let a = 0; a < this.pairCount; a++) {
        const i = this.pairI[a],
          j = this.pairJ[a],
          k = i * 3,
          b = j * 3;
        const dx = p[b] - p[k],
          dy = p[b + 1] - p[k + 1],
          dz = p[b + 2] - p[k + 2];
        let invR = this.pairInvR[a],
          q = this.pairQ[a];
        if (iteration > 0) {
          const r = Math.max(0.00001, Math.sqrt(dx * dx + dy * dy + dz * dz));
          q = Math.max(0, 1 - r / H);
          invR = 1 / r;
          this.pairQ[a] = q;
          this.pairInvR[a] = invR;
        }
        this.densities[i] += q * q;
        this.densities[j] += q * q;
        const g = (2 * q * invR) / (H * 3.6),
          gx = dx * g,
          gy = dy * g,
          gz = dz * g;
        this.gradient[k] += gx;
        this.gradient[k + 1] += gy;
        this.gradient[k + 2] += gz;
        this.gradient[b] -= gx;
        this.gradient[b + 1] -= gy;
        this.gradient[b + 2] -= gz;
        const square = gx * gx + gy * gy + gz * gz;
        this.gradientSquared[i] += square;
        this.gradientSquared[j] += square;
      }
      for (let i = 0; i < this.count; i++) {
        const k = i * 3,
          g = this.gradient;
        const denominator =
          this.gradientSquared[i] +
          g[k] * g[k] +
          g[k + 1] * g[k + 1] +
          g[k + 2] * g[k + 2] +
          2;
        this.pressure[i] =
          -Math.max(0, this.densities[i] / 3.6 - 1) / denominator;
      }
      for (let a = 0; a < this.pairCount; a++) {
        const i = this.pairI[a],
          j = this.pairJ[a],
          k = i * 3,
          b = j * 3;
        const factor =
          ((this.pressure[i] + this.pressure[j]) *
            2 *
            this.pairQ[a] *
            this.pairInvR[a]) /
          (H * 3.6);
        const dx = (p[b] - p[k]) * factor,
          dy = (p[b + 1] - p[k + 1]) * factor,
          dz = (p[b + 2] - p[k + 2]) * factor;
        corr[k] += dx;
        corr[k + 1] += dy;
        corr[k + 2] += dz;
        corr[b] -= dx;
        corr[b + 1] -= dy;
        corr[b + 2] -= dz;
      }
      for (let k = 0; k < this.count * 3; k += 3) {
        const length = Math.sqrt(
          corr[k] * corr[k] +
            corr[k + 1] * corr[k + 1] +
            corr[k + 2] * corr[k + 2],
        );
        const scale = length > 0.017 ? 0.017 / length : 1;
        p[k] = clamp(p[k] + corr[k] * scale, -HALF_X, HALF_X);
        p[k + 1] = clamp(p[k + 1] + corr[k + 1] * scale, FLOOR + 0.033, 3.8);
        p[k + 2] = clamp(p[k + 2] + corr[k + 2] * scale, -HALF_Z, HALF_Z);
      }
    }
    for (let k = 0; k < this.count * 3; k++)
      v[k] = ((p[k] - prev[k]) / dt) * 0.998;
    // XSPH-style velocity averaging; equal and opposite impulses damp relative motion.
    corr.fill(0);
    const viscosity = 0.002 + forces.viscosity * 0.065;
    for (let a = 0; a < this.pairCount; a++) {
      const k = this.pairI[a] * 3,
        b = this.pairJ[a] * 3;
      const q = this.pairQ[a];
      for (let axis = 0; axis < 3; axis++) {
        const change = (v[b + axis] - v[k + axis]) * q * q * viscosity;
        corr[k + axis] += change;
        corr[b + axis] -= change;
      }
    }
    for (let k = 0; k < this.count * 3; k++) v[k] += corr[k];
  }
}

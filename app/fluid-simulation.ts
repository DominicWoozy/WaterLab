/** Particle fluid with spatial hashing and double-density relaxation.
 * Based on Clavet, Beaudoin & Poulin (2005), without elastic springs.
 * Fixed timesteps and pairwise symmetric corrections preserve stable motion.
 */
export const CAPACITY = 2600;
export const FLOOR = -0.95;
export const HALF_X = 1.78;
export const HALF_Z = 1.28;
const H = 0.29;
const H2 = H * H;
const NX = 16,
  NY = 20,
  NZ = 13;
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
  private nearPressure = new Float32Array(CAPACITY);
  private heads = new Int32Array(NX * NY * NZ);
  private next = new Int32Array(CAPACITY);
  private cells = new Int32Array(CAPACITY * 3);
  private pairI = new Uint16Array(CAPACITY * 90);
  private pairJ = new Uint16Array(CAPACITY * 90);
  private pairQ = new Float32Array(CAPACITY * 90);
  private pairInvR = new Float32Array(CAPACITY * 90);
  private pairCount = 0;
  private seed = 71429;
  count = 0;
  time = 0;
  private shakeUntil = 0;

  constructor(count = 1700) {
    this.reset(count);
  }
  private random() {
    this.seed = (Math.imul(this.seed, 1664525) + 1013904223) >>> 0;
    return this.seed / 4294967296;
  }
  reset(count = 1700) {
    this.count = 0;
    this.time = 0;
    this.seed = 71429;
    this.shakeUntil = 0;
    this.velocities.fill(0);
    // Begin with a level basin, not an elastic column or a moving solid block.
    this.densities.fill(3.6);
    const amount = clamp(Math.round(count), 0, CAPACITY);
    const columns = 25,
      rows = 17;
    for (let i = 0; i < amount; i++) {
      const column = i % (columns * rows),
        layer = Math.floor(i / (columns * rows));
      const x = column % columns,
        z = Math.floor(column / columns);
      this.add(
        (x - (columns - 1) / 2) * 0.14,
        FLOOR + 0.08 + layer * 0.135,
        (z - (rows - 1) / 2) * 0.15,
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
  drain(amount = 250) {
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
  private buildPairs() {
    const p = this.positions;
    this.heads.fill(-1);
    for (let i = 0; i < this.count; i++) {
      const k = i * 3;
      const x = clamp(Math.floor((p[k] + 2.1) / H), 0, NX - 1);
      const y = clamp(Math.floor((p[k + 1] + 1.1) / H), 0, NY - 1);
      const z = clamp(Math.floor((p[k + 2] + 1.6) / H), 0, NZ - 1);
      this.cells[k] = x;
      this.cells[k + 1] = y;
      this.cells[k + 2] = z;
      const cell = x + NX * (y + NY * z);
      this.next[i] = this.heads[cell];
      this.heads[cell] = i;
    }
    this.pairCount = 0;
    for (let i = 0; i < this.count; i++) {
      const k = i * 3,
        cx = this.cells[k],
        cy = this.cells[k + 1],
        cz = this.cells[k + 2];
      for (let z = Math.max(0, cz - 1); z <= Math.min(NZ - 1, cz + 1); z++)
        for (let y = Math.max(0, cy - 1); y <= Math.min(NY - 1, cy + 1); y++)
          for (
            let x = Math.max(0, cx - 1);
            x <= Math.min(NX - 1, cx + 1);
            x++
          ) {
            for (
              let j = this.heads[x + NX * (y + NY * z)];
              j !== -1;
              j = this.next[j]
            ) {
              if (j <= i) continue;
              const b = j * 3,
                dx = p[b] - p[k],
                dy = p[b + 1] - p[k + 1],
                dz = p[b + 2] - p[k + 2];
              const r2 = dx * dx + dy * dy + dz * dz;
              if (r2 < H2 && this.pairCount < this.pairI.length) {
                const r = Math.max(0.00001, Math.sqrt(r2));
                this.pairQ[this.pairCount] = 1 - r / H;
                this.pairInvR[this.pairCount] = 1 / r;
                this.pairI[this.pairCount] = i;
                this.pairJ[this.pairCount++] = j;
              }
            }
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
      const speed = Math.hypot(v[k], v[k + 1], v[k + 2]);
      const limit = speed > 12 ? 12 / speed : 1;
      p[k] += v[k] * limit * dt;
      p[k + 1] += v[k + 1] * limit * dt;
      p[k + 2] += v[k + 2] * limit * dt;
    }
    for (let iteration = 0; iteration < 2; iteration++) {
      this.buildPairs();
      this.pressure.fill(0);
      this.nearPressure.fill(0);
      corr.fill(0);
      for (let a = 0; a < this.pairCount; a++) {
        const i = this.pairI[a],
          j = this.pairJ[a];
        const q = this.pairQ[a];
        this.pressure[i] += q * q;
        this.pressure[j] += q * q;
        this.nearPressure[i] += q * q * q;
        this.nearPressure[j] += q * q * q;
      }
      for (let i = 0; i < this.count; i++) {
        this.densities[i] = this.pressure[i];
        // Compression-only pressure avoids artificial tensile attraction (the jelly effect).
        const compression = Math.max(0, this.pressure[i] - 3.6);
        this.pressure[i] = 950 * compression;
        this.nearPressure[i] *= Math.min(1, compression) * 180;
      }
      for (let a = 0; a < this.pairCount; a++) {
        const i = this.pairI[a],
          j = this.pairJ[a],
          k = i * 3,
          b = j * 3;
        let dx = p[b] - p[k],
          dy = p[b + 1] - p[k + 1],
          dz = p[b + 2] - p[k + 2];
        const q = this.pairQ[a];
        const displacement =
          clamp(
            dt *
              dt *
              ((this.pressure[i] + this.pressure[j]) * q +
                (this.nearPressure[i] + this.nearPressure[j]) * q * q) *
              0.25,
            -0.006,
            0.012,
          ) * this.pairInvR[a];
        dx *= displacement;
        dy *= displacement;
        dz *= displacement;
        corr[k] -= dx;
        corr[k + 1] -= dy;
        corr[k + 2] -= dz;
        corr[b] += dx;
        corr[b + 1] += dy;
        corr[b + 2] += dz;
      }
      for (let k = 0; k < this.count * 3; k += 3) {
        p[k] = clamp(p[k] + corr[k], -HALF_X, HALF_X);
        p[k + 1] = clamp(p[k + 1] + corr[k + 1], FLOOR + 0.06, 3.8);
        p[k + 2] = clamp(p[k + 2] + corr[k + 2], -HALF_Z, HALF_Z);
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

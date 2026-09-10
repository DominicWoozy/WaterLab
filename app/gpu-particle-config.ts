export const GPU_CAPACITY = 30000;
export const GPU_DEFAULT_COUNT = 15000;
export const PARTICLE_WIDTH = 256;
export const PARTICLE_HEIGHT = 128;
export const PARTICLE_QUALITIES = [15000, 30000, 50000, 70000, 100000] as const;
export type ParticleQuality = (typeof PARTICLE_QUALITIES)[number];

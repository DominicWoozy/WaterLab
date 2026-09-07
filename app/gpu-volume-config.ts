/** Nearly cubic voxels resolve smaller spray while preserving the bulk smoothing radius. */
export const GPU_VOLUME_SIZE = [128, 160, 96] as const;
export const GPU_ATLAS_SIZE = [1024, 1920] as const;
export const GPU_SLICES_PER_PARTICLE = 20;
export const BULK_KERNEL_RADIUS = 0.19;
export const SPRAY_KERNEL_RADIUS = 0.1;

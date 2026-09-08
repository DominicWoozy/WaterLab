import type { WaterSettings } from './water-engine';
type Tool = {
  name: string;
  title: string;
  description: string;
  inputSchema: object;
  annotations: { readOnlyHint: boolean; untrustedContentHint: boolean };
  execute: (input: unknown) => unknown;
};
type Context = {
  registerTool: (
    tool: Tool,
    options: { signal: AbortSignal },
  ) => void | Promise<void>;
};
export function registerWaterTools(
  get: () => WaterSettings,
  apply: (next: WaterSettings) => void,
  ripple: () => void,
) {
  const context = (document as Document & { modelContext?: Context })
    .modelContext;
  if (!context?.registerTool) return;
  const controller = new AbortController();
  const tools: Tool[] = [
    {
      name: 'configure_water',
      title: '调整水体',
      description:
        '调整当前水体的搅动力度、黏性、重力、时间速度、光照、反射、焦散或暂停状态，返回生效参数。',
      inputSchema: {
        type: 'object',
        properties: {
          strength: { type: 'number', minimum: 0.2, maximum: 2.5 },
          speed: { type: 'number', minimum: 0.25, maximum: 1.5 },
          viscosity: { type: 'number', minimum: 0, maximum: 1 },
          gravity: { type: 'number', minimum: 0, maximum: 14 },
          agitation: { type: 'number', minimum: 0, maximum: 1.4 },
          particles: { type: 'boolean' },
          details: { type: 'boolean' },
          light: { type: 'number', minimum: 0.2, maximum: 2.5 },
          reflection: { type: 'boolean' },
          caustics: { type: 'boolean' },
          paused: { type: 'boolean' },
        },
        additionalProperties: false,
      },
      annotations: { readOnlyHint: false, untrustedContentHint: false },
      execute: async (input) => {
        if (!input || typeof input !== 'object' || Array.isArray(input))
          throw new Error('参数必须是对象');
        const next = { ...get() };
        const bounds: Record<string, number[]> = {
          strength: [0.2, 2.5],
          speed: [0.25, 1.5],
          viscosity: [0, 1],
          gravity: [0, 14],
          agitation: [0, 1.4],
          light: [0.2, 2.5],
        };
        for (const [key, value] of Object.entries(input)) {
          if (key in bounds) {
            const [min, max] = bounds[key];
            if (
              typeof value !== 'number' ||
              !Number.isFinite(value) ||
              value < min ||
              value > max
            )
              throw new Error(`${key} 超出允许范围`);
          } else if (
            [
              'reflection',
              'caustics',
              'paused',
              'particles',
              'details',
            ].includes(key)
          ) {
            if (typeof value !== 'boolean')
              throw new Error(`${key} 必须是布尔值`);
          } else throw new Error(`未知参数 ${key}`);
          Object.assign(next, { [key]: value });
        }
        apply(next);
        await new Promise<void>((resolve) =>
          requestAnimationFrame(() => resolve()),
        );
        return get();
      },
    },
    {
      name: 'add_water_ripple',
      title: '添加涟漪',
      description:
        '在当前水体中央施加一次涟漪扰动。暂停时须恢复模拟才能看到波纹传播。',
      inputSchema: {
        type: 'object',
        properties: {},
        additionalProperties: false,
      },
      annotations: { readOnlyHint: false, untrustedContentHint: false },
      execute: (input) => {
        if (
          !input ||
          typeof input !== 'object' ||
          Array.isArray(input) ||
          Object.keys(input).length
        )
          throw new Error('此操作不接受参数');
        ripple();
        return { added: true, paused: get().paused };
      },
    },
  ];
  for (const tool of tools) {
    try {
      void Promise.resolve(
        context.registerTool(tool, { signal: controller.signal }),
      ).catch(() => {});
    } catch {
      /* Optional API is not available in every browser. */
    }
  }
  return () => controller.abort();
}

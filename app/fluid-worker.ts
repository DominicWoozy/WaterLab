import { FluidRuntime, type FluidJob } from './fluid-runtime.ts';
const runtime = new FluidRuntime();
const context = self as unknown as {
  onmessage: ((event: MessageEvent<FluidJob>) => void) | null;
  postMessage: (message: unknown, transfer: Transferable[]) => void;
};
context.onmessage = (event) => {
  try {
    const frame = runtime.run(event.data);
    const transfer: Transferable[] = [];
    if (frame.volume) transfer.push(frame.volume);
    if (frame.positions) transfer.push(frame.positions);
    context.postMessage({ type: 'frame', ...frame }, transfer);
  } catch (error) {
    context.postMessage(
      {
        type: 'error',
        message: error instanceof Error ? error.message : '后台流体计算失败',
      },
      [],
    );
  }
};

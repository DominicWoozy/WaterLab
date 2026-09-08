/** Upload only immutable model assets; simulated state is never read back. */
export function loadDuckModel(
  gl: WebGL2RenderingContext,
  onError: (message: string) => void,
) {
  const textures: WebGLTexture[] = [];
  let ready = false,
    disposed = false;
  const abort = new AbortController();
  const texture = () => {
    const t = gl.createTexture()!;
    textures.push(t);
    gl.bindTexture(gl.TEXTURE_2D, t);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texImage2D(
      gl.TEXTURE_2D,
      0,
      gl.RGBA32F,
      1,
      1,
      0,
      gl.RGBA,
      gl.FLOAT,
      new Float32Array(4),
    );
    return t;
  };
  const bvh = texture(),
    triangles = texture(),
    albedo = texture();
  const fetchAsset = async (file: string) => {
    const response = await fetch(`/models/duck/${file}`, {
      signal: abort.signal,
    });
    if (!response.ok) throw new Error(`模型资源加载失败 (${response.status})`);
    return response;
  };
  (async () => {
    const [meta, bvhBytes, triangleBytes, blob] = await Promise.all([
      fetchAsset('mesh.json').then((r) => r.json()) as Promise<{
        width: number;
        bvhHeight: number;
        triangleHeight: number;
      }>,
      fetchAsset('bvh.bin').then((r) => r.arrayBuffer()),
      fetchAsset('triangles.bin').then((r) => r.arrayBuffer()),
      fetchAsset('DuckCM.png').then((r) => r.blob()),
    ]);
    const bitmap = await createImageBitmap(blob, {
      imageOrientation: 'none',
      premultiplyAlpha: 'none',
      colorSpaceConversion: 'none',
    });
    if (disposed) {
      bitmap.close();
      return;
    }
    for (const [target, bytes, height] of [
      [bvh, bvhBytes, meta.bvhHeight],
      [triangles, triangleBytes, meta.triangleHeight],
    ] as const) {
      if (bytes.byteLength !== meta.width * height * 16)
        throw new Error('模型网格数据不完整');
      gl.bindTexture(gl.TEXTURE_2D, target);
      gl.texImage2D(
        gl.TEXTURE_2D,
        0,
        gl.RGBA32F,
        meta.width,
        height,
        0,
        gl.RGBA,
        gl.FLOAT,
        new Float32Array(bytes),
      );
    }
    gl.bindTexture(gl.TEXTURE_2D, albedo);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, bitmap);
    gl.texParameteri(
      gl.TEXTURE_2D,
      gl.TEXTURE_MIN_FILTER,
      gl.LINEAR_MIPMAP_LINEAR,
    );
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.generateMipmap(gl.TEXTURE_2D);
    bitmap.close();
    ready = true;
  })().catch((error) => {
    if (!disposed)
      onError(
        error instanceof Error
          ? error.message
          : '小鸭子模型加载失败，请刷新重试。',
      );
  });
  return {
    bvh,
    triangles,
    albedo,
    get ready() {
      return ready;
    },
    destroy() {
      disposed = true;
      abort.abort();
      textures.forEach((t) => gl.deleteTexture(t));
    },
  };
}

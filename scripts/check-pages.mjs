import { readFileSync, existsSync, writeFileSync } from 'node:fs';
import assert from 'node:assert/strict';
import { join } from 'node:path';
const root = 'dist/pages';
const base = process.env.PAGES_BASE_PATH || '';
const html = readFileSync(join(root, 'index.html'), 'utf8');
assert.ok(html.includes('水体实验室'), 'Static HTML was not rendered');
const paths = [...html.matchAll(/(?:src|href)="([^"#?]+)(?:[?#][^"]*)?"/g)]
  .map((match) => match[1])
  .filter((path) => path.startsWith('/') && !path.startsWith('//'));
for (const path of paths) {
  assert.ok(
    path.startsWith(`${base}/`),
    `Asset escapes the Pages base path: ${path}`,
  );
  assert.ok(
    existsSync(join(root, path.slice(base.length))),
    `Missing build asset: ${path}`,
  );
}
for (const asset of [
  'mesh.json',
  'bvh.bin',
  'triangles.bin',
  'DuckCM.png',
  'Duck.glb',
  'LICENSE.txt',
])
  assert.ok(
    existsSync(join(root, 'models/duck', asset)),
    `Missing duck asset: ${asset}`,
  );
assert.ok(
  !existsSync(join(root, 'server')),
  'Server bundle must not be published',
);
writeFileSync(join(root, '.nojekyll'), '');
console.log(
  `Static page and ${paths.length} HTML asset paths verified for ${base || '/'}`,
);

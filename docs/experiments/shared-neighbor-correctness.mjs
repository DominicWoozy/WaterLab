// Reuse the independent all-pairs oracle with only the experimental dispatch hook.
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { execFileSync } from 'node:child_process';
const mode = process.env.MODE || 'shared';
const hook = mode === 'shared' ? 'sharedNeighbors' : 'neighborWorkgroups';
const moduleURL = new URL(
  mode === 'shared' ? './shared-neighbors.mjs' : './neighbor-workgroups.mjs',
  import.meta.url,
);
let source = readFileSync(
  new URL('../../tests/webgpu-neighbors.mjs', import.meta.url),
  'utf8',
);
const marker = 'const kept =';
assert.ok(source.includes(marker), 'reference fixture injection point changed');
source = source.replaceAll(
  "'../app/",
  `'${new URL('../../app/', import.meta.url).href}`,
);
source = source.replace(
  marker,
  `const {${hook}} = await import(${JSON.stringify(moduleURL.href)});\nawait ${hook}(sim, ${Number(process.env.WIDTH || 32)}, ${process.env.SHARED !== '0'});\n${marker}`,
);
execFileSync(
  process.execPath,
  ['--experimental-strip-types', '--input-type=module', '--eval', source],
  { stdio: 'inherit', cwd: new URL('../../', import.meta.url) },
);

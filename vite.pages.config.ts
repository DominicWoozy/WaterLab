import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import tailwindcss from '@tailwindcss/postcss';
import { fileURLToPath } from 'node:url';

// A static entry reuses the existing client page without requiring a Workers server.
export default defineConfig({
  base: `${process.env.PAGES_BASE_PATH || ''}/`,
  define: {
    'process.env.NEXT_PUBLIC_ASSET_BASE': JSON.stringify(
      process.env.PAGES_BASE_PATH || '',
    ),
  },
  resolve: { alias: { '@': fileURLToPath(new URL('.', import.meta.url)) } },
  css: { postcss: { plugins: [tailwindcss()] } },
  plugins: [react()],
  build: { outDir: 'dist/pages', emptyOutDir: true },
});

import { defineConfig } from 'vite';

// BASE_PATH is set by the GitHub Actions workflow (/<repo-name>/).
// Locally it falls back to '/cps/'; use BASE_PATH=/ for a root/custom domain.
export default defineConfig({
  base: process.env.BASE_PATH ?? '/cps/',
  // GitHub Pages serves the committed ../docs folder
  build: { outDir: '../docs', emptyOutDir: true, assetsInlineLimit: 0 },
});

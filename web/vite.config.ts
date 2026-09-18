import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    // during development the API runs separately on 8000
    proxy: { "/api": "http://127.0.0.1:8000", "/healthz": "http://127.0.0.1:8000" },
  },
  build: {
    outDir: "dist",
    rollupOptions: {
      output: {
        // three.js is large and only needed once the viewer mounts
        manualChunks: { three: ["three", "@react-three/fiber", "@react-three/drei"] },
      },
    },
  },
});

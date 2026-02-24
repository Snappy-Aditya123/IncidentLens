import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";
import path from "path";

const BACKEND_URL = process.env.VITE_BACKEND_URL ?? "http://localhost:8000";
const BACKEND_WS = BACKEND_URL.replace(/^http/, "ws");

export default defineConfig({
  plugins: [react(), tailwindcss()],
  root: ".",
  publicDir: "public",
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "app"),
    },
  },
  server: {
    port: 5173,
    proxy: {
      "/api": {
        target: BACKEND_URL,
        changeOrigin: true,
      },
      "/ws": {
        target: BACKEND_WS,
        ws: true,
      },
      "/health": {
        target: BACKEND_URL,
        changeOrigin: true,
      },
    },
  },
});

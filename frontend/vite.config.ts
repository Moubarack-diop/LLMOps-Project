import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// En dev, les appels API sont proxifiés vers FastAPI (port 8000) :
// pas de CORS à gérer et les mêmes URLs relatives fonctionnent en prod
// (le build est servi par FastAPI sur la même origine).
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/health": "http://localhost:8000",
      "/notes": "http://localhost:8000",
      "/query": "http://localhost:8000",
      "/metrics": "http://localhost:8000",
      "/ingest": "http://localhost:8000",
    },
  },
});

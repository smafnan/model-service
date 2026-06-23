import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  base: "./",
  server: {
    // Proxy the model-service endpoints to the FastAPI backend during dev.
    proxy: {
      "/predict": "http://localhost:8000",
      "/health": "http://localhost:8000",
      "/info": "http://localhost:8000",
    },
  },
});

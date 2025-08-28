// next.config.ts
import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Expose the API base directly to the browser; point this to your FastAPI URL.
  // In docker-compose, set NEXT_PUBLIC_API_BASE_URL=http://localhost:8080
  env: {
    NEXT_PUBLIC_API_BASE_URL:
      process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost:8080",
  },

  // IMPORTANT: remove rewrites/proxying to the API.
  // The proxy is what’s closing the socket on slow first calls (Ollama warmup).
  // If you had rewrites() before, delete it entirely.
};

export default nextConfig;

import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  async rewrites() {
    // At runtime, the browser hits http://localhost:3000/backend/*
    // Next.js (inside the UI container) will proxy to http://api:8080/*
    return [
      {
        source: "/backend/:path*",
        destination: "http://api:8080/:path*",
      },
    ];
  },
};

export default nextConfig;

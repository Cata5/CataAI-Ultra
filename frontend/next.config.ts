import type { NextConfig } from "next";

const BACKEND_URL = process.env.NEXT_PUBLIC_BACKEND_URL || "http://127.0.0.1:8081";
const isTauri = process.env.TAURI_ENV_PLATFORM !== undefined || process.env.TAURI_BUILD === "true";

const nextConfig: NextConfig = {
  reactStrictMode: false,
  output: isTauri ? "export" : undefined,
  trailingSlash: false,
  images: {
    unoptimized: true,
  },
  turbopack: {
    rules: {
      "*.css": {
        loaders: ["@tailwindcss/turbopack"],
        as: "*.css",
      },
    },
  },
  async rewrites() {
    if (isTauri) return [];
    return [
      {
        source: "/api/:path*",
        destination: `${BACKEND_URL}/api/:path*`,
      },
    ];
  },
};

export default nextConfig;

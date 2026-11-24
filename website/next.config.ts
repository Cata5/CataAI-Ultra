import type { NextConfig } from "next";

// GitHub Pages serves project sites from /<repo>/. The deploy workflow sets this
// automatically; an empty value works for a user site or a custom domain.
const basePath = process.env.NEXT_PUBLIC_BASE_PATH ?? "";

const nextConfig: NextConfig = {
  output: "export",
  trailingSlash: true,
  images: { unoptimized: true },
  basePath,
  assetPrefix: basePath || undefined,
};

export default nextConfig;
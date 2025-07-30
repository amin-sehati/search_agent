/** @type {import('next').NextConfig} */
const nextConfig = {
  // Remove env variables from here - they should be set in Vercel dashboard
  // or as environment variables directly
  experimental: {
    serverComponentsExternalPackages: ['python-shell'],
  },
  // Optimize for Vercel deployment
  output: 'standalone',
}

module.exports = nextConfig
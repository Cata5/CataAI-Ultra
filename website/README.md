# LLM Maker website

Static marketing/dossier site for LLM Maker, built with Next.js and exported for
GitHub Pages. Content reflects the repository audit and the professional-grade
roadmap (status as of the audit follow-up).

## Local development

From the repository root:

```bash
cd website
npm ci
npm run dev
```

Open http://localhost:3000.

## Build

```bash
npm run build
```

The static site is written to `out/`. For a GitHub Pages project site, build with
the repository base path:

```bash
NEXT_PUBLIC_BASE_PATH=/CataAI-Ultra npm run build
```

## Deploy

`.github/workflows/pages.yml` in the repository root publishes `website/out` on
every push to `main` or `experimental` that touches this folder.

**Required repository setting:** Settings -> Pages -> Build and deployment ->
Source = **GitHub Actions**. If the source stays on "Deploy from a branch",
GitHub ignores the workflow and serves the root `README.md` through Jekyll. That
is the Jekyll/README page, not this site. Keep only `pages.yml` as the deploy
workflow; a duplicate workflow causes competing deployments.

The workflow uses `actions/configure-pages` to supply the base path, including
the empty base path used by `<owner>.github.io` user sites and custom domains.

No analytics, no cookies, no external requests at runtime.

## Changelog

`lib/changelog.ts` is the single source for the `/changelog/` page. The release
tab lists the public `main` history. The experimental tab lists only commits
that are on `experimental` and not on `main`, so the two tabs never repeat the
same change. Add new entries there when the branch history moves.

The sidebar theme toggle stores the choice in `localStorage` as
`llmmaker-theme`. A small inline script applies that choice before hydration;
`suppressHydrationWarning` on `<html>` covers the intentional theme attribute
change, so React does not report a hydration mismatch.

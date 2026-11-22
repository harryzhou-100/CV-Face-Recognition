# Quantifying Kindness — Project Website

A gallery-quality presentation website for the *Quantifying Kindness* performance art
and computer vision installation, with a **live real-time kindness demo** running
entirely in-browser via face-api.js.

---

## Running Locally

The site is a single static HTML file — no build step required.

### Option 1: Python (simplest)

```bash
cd website
python3 -m http.server 8080
```

Open: http://localhost:8080

### Option 2: Node / npx

```bash
cd website
npx serve .
```

Open: http://localhost:3000

### Option 3: VS Code Live Server

Right-click `website/index.html` → **Open with Live Server**

---

## What's Inside

| Section | Description |
|---|---|
| **Hero** | Full-screen title with the core provocation |
| **Concept** | Artist statement, expression→score mapping table, stat cards |
| **How It Works** | Four-step process from artist stillness to donation trigger |
| **Live Demo** | Real-time webcam kindness index via face-api.js (in-browser, no server) |
| **Crowd Visualization** | Generative canvas animation — kindness rippling through a crowd |
| **Inspiration** | Marina Abramović's *The Artist is Present*, with animated stillness canvas |
| **Artifact** | Mock post-exhibition emotional trajectory visualization |
| **Footer** | Navigation + technical credits |

---

## Live Demo Notes

- Uses **face-api.js v1.7.12** (TinyFaceDetector + FaceExpressionNet) via CDN
- Models (~2 MB) load from `@vladmandic/face-api` on jsDelivr — requires internet
- **No data leaves your device**: all processing is local to your browser
- Expression → score mapping:
  - `happy` → 8–10
  - `surprised` → 6–8
  - `neutral` → 5
  - `fearful` → 3–5
  - `sad` → 2–4
  - `angry` / `disgusted` → 1–3
- When score ≥ 7, a micro-donation counter increments and an ambient glow triggers
- Score uses a 70/30 exponential moving average for smooth display

---

## Technical Stack

- **Framework**: Vanilla HTML/CSS/JavaScript (no build toolchain)
- **CV Library**: [face-api.js](https://github.com/vladmandic/face-api) via CDN
- **Canvases**: Three generative animations (crowd ripple, stillness rings, trajectory arcs)
- **Fonts**: System serif + sans (no external font requests)
- **Deployment**: Any static host (GitHub Pages, Netlify, Vercel, etc.)

---

## Deploying to GitHub Pages

```bash
# From repo root on the website-demo branch
git subtree push --prefix website origin gh-pages
```

Or set GitHub Pages source to `website-demo` branch, `website/` folder.

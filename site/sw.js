// Cache-first storage of map data (terrain, imagery, 3D tiles) and the CesiumJS library, so a repeat
// visit loads from disk instead of the network. VERSION is replaced with the commit SHA on deploy
// (.github/workflows/pages.yml); a new deploy starts a fresh cache and deletes the old one.
const VERSION = "__VERSION__";
const CACHE = `fnv-maps-${VERSION}`;

self.addEventListener("install", () => self.skipWaiting());

self.addEventListener("activate", (event) => {
  event.waitUntil((async () => {
    for (const key of await caches.keys()) {
      if (key.startsWith("fnv-maps-") && key !== CACHE) await caches.delete(key);
    }
    await self.clients.claim();
  })());
});

// map data published from separate repositories (maps.json "data"), served from the same github.io origin
const DATA_REPOS = ["/tes-skyrim/"];

function cacheable(url) {
  if (url.origin === self.location.origin) {
    return url.pathname.includes("/maps/") || DATA_REPOS.some((p) => url.pathname.startsWith(p));
  }
  return url.hostname === "cdn.jsdelivr.net" && url.pathname.startsWith("/npm/cesium@");
}

self.addEventListener("fetch", (event) => {
  const req = event.request;
  if (req.method !== "GET" || !cacheable(new URL(req.url))) return;
  event.respondWith((async () => {
    const cache = await caches.open(CACHE);
    const hit = await cache.match(req, { ignoreVary: true });
    if (hit) return hit;
    const res = await fetch(req);
    if (res.ok && res.status === 200) cache.put(req, res.clone()).catch(() => {});   // quota full: just skip
    return res;
  })());
});

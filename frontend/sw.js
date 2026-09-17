/*
 * WildCast service worker: caches the static app shell (HTML/CSS/JS/icons/
 * manifest) so the app installs as a real PWA and opens instantly offline,
 * WITHOUT ever caching API responses. Predictions must always be a live
 * network call -- caching them would risk showing stale, wrongly-labeled
 * (real vs. synthetic-demo) data days after it was fetched, which directly
 * conflicts with the "label real vs. demo data clearly" requirement this
 * app is built around. Bump CACHE_NAME on every deploy that changes any
 * cached file so old clients pick up the new shell instead of a stale one.
 */
const CACHE_NAME = "wildcast-shell-v1";
const APP_SHELL = [
  "./",
  "./index.html",
  "./manifest.json",
  "./icons/icon-192.png",
  "./icons/icon-512.png",
  "./icons/icon-512-maskable.png",
  "./icons/apple-touch-icon.png",
  "./icons/favicon-32.png",
  "./icons/favicon-16.png",
];

self.addEventListener("install", (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME).then((cache) => cache.addAll(APP_SHELL)).then(() => self.skipWaiting())
  );
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    caches.keys().then((names) =>
      Promise.all(names.filter((name) => name !== CACHE_NAME).map((name) => caches.delete(name)))
    ).then(() => self.clients.claim())
  );
});

self.addEventListener("fetch", (event) => {
  const req = event.request;
  if (req.method !== "GET") return;

  const url = new URL(req.url);
  // Only ever cache same-origin app-shell files (this page, its CSS/JS,
  // icons, manifest). Cross-origin requests -- the WildCast API (any
  // backend host the user configured), the Leaflet CDN, OpenStreetMap map
  // tiles -- always go straight to the network: predictions must be live,
  // and third-party assets are already CDN-cached by the browser itself.
  if (url.origin !== self.location.origin) return;

  event.respondWith(
    caches.match(req).then((cached) => {
      const network = fetch(req)
        .then((resp) => {
          if (resp && resp.ok) {
            const copy = resp.clone();
            caches.open(CACHE_NAME).then((cache) => cache.put(req, copy));
          }
          return resp;
        })
        .catch(() => cached); // offline: fall back to whatever shell version is cached
      return cached || network;
    })
  );
});

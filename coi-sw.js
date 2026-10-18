// Cross-origin isolation for the trainer page only. GitHub Pages cannot send response headers, so this service worker
// adds them. With them, the browser reader runs on several CPU threads: about 7 s per action instead of 28 s (measured
// with 8 threads, Chromium, 12 AgentDojo steps). "credentialless" keeps cross-origin fonts and model files loading;
// browsers without it (Safari) stay single-threaded and still work. No request is changed otherwise; nothing is cached.
self.addEventListener('install', () => self.skipWaiting());
self.addEventListener('activate', ev => ev.waitUntil(self.clients.claim()));
self.addEventListener('fetch', ev => {
  const req = ev.request, url = new URL(req.url);
  if (url.origin !== self.location.origin) return;
  const page = req.mode === 'navigate' && url.pathname.endsWith('/train.html');
  const worker = req.destination === 'worker' || req.destination === 'sharedworker';
  if (!page && !worker) return;
  ev.respondWith(fetch(req).then(res => {
    if (res.status === 0 || res.type === 'opaque') return res;
    const h = new Headers(res.headers);
    h.set('Cross-Origin-Opener-Policy', 'same-origin');
    h.set('Cross-Origin-Embedder-Policy', 'credentialless');
    return new Response(res.body, { status: res.status, statusText: res.statusText, headers: h });
  }));
});

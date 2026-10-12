// One place for calls to the official FPL API.
//
// The bootstrap file is about 1.5 MB and every page view used to fetch it two or
// three times (projections overlay, fixtures, team import). It changes a few
// times a day, so a short in-memory cache per server instance removes almost all
// of that traffic without serving stale injury news for more than a minute.

const USER_AGENT = "FPL-Lens/8.0";
const cache = new Map<string, { expires: number; value: Promise<unknown> }>();

export async function officialJson<T>(url: string, ttlMs = 0): Promise<T> {
  const now = Date.now();
  const hit = ttlMs > 0 ? cache.get(url) : undefined;
  if (hit && hit.expires > now) return hit.value as Promise<T>;
  const value = fetch(url, {
    headers: { "User-Agent": USER_AGENT },
    cache: "no-store",
    signal: AbortSignal.timeout(8_000),
  }).then((response) => {
    if (!response.ok) throw new Error(`Official FPL API returned ${response.status}`);
    return response.json() as Promise<T>;
  });
  if (ttlMs > 0) {
    cache.set(url, { expires: now + ttlMs, value });
    // A failed call must not be cached for the whole window.
    value.catch(() => cache.delete(url));
  }
  return value;
}

export const BOOTSTRAP_URL = "https://fantasy.premierleague.com/api/bootstrap-static/";
export const BOOTSTRAP_TTL_MS = 60_000;

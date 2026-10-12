"use client";

import { useCallback, useEffect, useMemo, useState, useSyncExternalStore } from "react";
import results from "../data/model-results.json";
import { Captains } from "./Captains";
import { countdown } from "./format";
import { Fixtures } from "./Fixtures";
import { MyTeam } from "./MyTeam";
import { ComparePanel, PlayerDetail } from "./PlayerDetail";
import { Planner } from "./Planner";
import { PlayersTable } from "./PlayersTable";
import { ThisWeek } from "./ThisWeek";
import { TrackRecord } from "./TrackRecord";
import type { ImportedTeam, Player } from "./types";

const TABS = [
  { id: "week", label: "This week" },
  { id: "players", label: "Players" },
  { id: "captain", label: "Captain" },
  { id: "fixtures", label: "Fixtures" },
  { id: "team", label: "My team" },
  { id: "planner", label: "Wildcard planner" },
  { id: "record", label: "Track record" },
] as const;

type TabId = (typeof TABS)[number]["id"];

// The open tab lives in the URL hash, so tabs can be bookmarked and shared.
function subscribeHash(onChange: () => void) {
  window.addEventListener("hashchange", onChange);
  return () => window.removeEventListener("hashchange", onChange);
}
const readHash = () => window.location.hash.replace("#", "");
const serverHash = () => "";

export default function FplDashboard() {
  const hash = useSyncExternalStore(subscribeHash, readHash, serverHash);
  const tab = (TABS.find((item) => item.id === hash)?.id ?? "week") as TabId;
  const [now, setNow] = useState(() => Date.now());
  const [pool, setPool] = useState<Player[]>(() => {
    const initial = [...(results.squad as unknown as Player[]), ...(results.watchlist as unknown as Player[])];
    return [...new Map(initial.map((player) => [player.id, player])).values()];
  });
  const [poolSource, setPoolSource] = useState<"loading" | "live" | "artifact">("loading");
  const [openId, setOpenId] = useState<number | null>(null);
  const [compareIds, setCompareIds] = useState<number[]>([]);
  const [team, setTeam] = useState<ImportedTeam | null>(null);

  useEffect(() => {
    const timer = window.setInterval(() => setNow(Date.now()), 30_000);
    return () => window.clearInterval(timer);
  }, []);

  useEffect(() => {
    const controller = new AbortController();
    fetch("/api/projections?limit=1000&view=app", { signal: controller.signal })
      .then((response) => {
        if (!response.ok) throw new Error(`Projection API returned ${response.status}`);
        return response.json() as Promise<{ players: Player[]; availabilitySource: string }>;
      })
      .then((payload) => {
        setPool(payload.players);
        setPoolSource(payload.availabilitySource === "live-official-fpl" ? "live" : "artifact");
      })
      .catch((error: unknown) => {
        if (!(error instanceof DOMException && error.name === "AbortError")) setPoolSource("artifact");
      });
    return () => controller.abort();
  }, []);

  // A saved team ID is loaded once, quietly, so My team and the captain filter work straight away.
  useEffect(() => {
    let saved = "";
    try {
      saved = window.localStorage.getItem("fpl-lens-entry") ?? "";
    } catch {
      return;
    }
    if (!/^\d+$/.test(saved)) return;
    const controller = new AbortController();
    fetch(`/api/team/${saved}`, { signal: controller.signal })
      .then((response) => (response.ok ? (response.json() as Promise<ImportedTeam>) : null))
      .then((payload) => payload && setTeam(payload))
      .catch(() => undefined);
    return () => controller.abort();
  }, []);

  const byId = useMemo(() => new Map(pool.map((player) => [player.id, player])), [pool]);
  const open = useCallback((player: Player) => setOpenId(player.id), []);
  const toggleCompare = useCallback((player: Player) => {
    setCompareIds((ids) =>
      ids.includes(player.id) ? ids.filter((id) => id !== player.id) : ids.length >= 4 ? ids : [...ids, player.id],
    );
  }, []);
  const opened = openId === null ? null : byId.get(openId) ?? null;
  const compared = compareIds.map((id) => byId.get(id)).filter((player): player is Player => Boolean(player));
  const ownedIds = team?.owned.map((pick) => pick.element) ?? [];
  const ownedTeams = [...new Set(team?.owned.map((pick) => pick.player.team) ?? [])];

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <span className="brand-mark" aria-hidden>FL</span>
          <span>FPL Lens</span>
        </div>
        <div className="deadline">
          <span>GW{results.headline.gameweek} deadline</span>
          <strong>{countdown(results.headline.deadline, now)}</strong>
        </div>
        <div className="freshness">
          <span>Projections {new Date(results.generatedAt).toLocaleDateString("en-GB", { day: "numeric", month: "short" })}</span>
          <span className={`dot ${poolSource}`} title={poolSource === "live" ? "Injury news is live from FPL" : poolSource === "loading" ? "Loading" : "Using saved injury news"} />
          <span>{poolSource === "live" ? "live news" : poolSource === "loading" ? "loading…" : "saved news"}</span>
        </div>
      </header>

      <nav className="tabs" aria-label="Sections">
        {TABS.map((item) => (
          <a
            key={item.id}
            href={`#${item.id}`}
            className={tab === item.id ? "active" : ""}
            aria-current={tab === item.id ? "page" : undefined}
            onClick={() => window.scrollTo({ top: 0 })}
          >
            {item.label}
            {item.id === "team" && team && <span className="tab-note">{team.manager.teamName}</span>}
          </a>
        ))}
      </nav>

      <main className="content">
        {tab === "week" && <ThisWeek pool={pool} onOpen={open} />}
        {tab === "players" && <PlayersTable pool={pool} onOpen={open} compareIds={compareIds} onToggleCompare={toggleCompare} />}
        {tab === "captain" && <Captains pool={pool} ownedIds={ownedIds} onOpen={open} />}
        {tab === "fixtures" && <Fixtures highlightTeams={ownedTeams} />}
        {tab === "team" && <MyTeam pool={pool} team={team} onTeam={setTeam} onOpen={open} />}
        {tab === "planner" && <Planner pool={pool} team={team} onOpen={open} />}
        {tab === "record" && <TrackRecord />}
      </main>

      <ComparePanel
        players={compared}
        onRemove={toggleCompare}
        onClear={() => setCompareIds([])}
        onOpen={open}
      />

      {opened && (
        <PlayerDetail
          player={opened}
          onClose={() => setOpenId(null)}
          inCompare={compareIds.includes(opened.id)}
          onToggleCompare={toggleCompare}
        />
      )}

      <footer className="footer">
        Projections are estimates, not guarantees. Check team news before every deadline. Data from the official FPL API.
      </footer>
    </div>
  );
}

import { useMemo, useState, type FormEvent } from "react";
import results from "../data/model-results.json";
import { availabilityIssue, bestLineup, fixtureLabel, money, one, signed } from "./format";
import { Pitch } from "./Pitch";
import type { ImportedTeam, Player } from "./types";
import { Card, Empty, Flag, PositionTag, Stat } from "./ui";

const CHIPS: Array<[string, string]> = [
  ["wildcard", "Wildcard"],
  ["freehit", "Free Hit"],
  ["bboost", "Bench Boost"],
  ["3xc", "Triple Captain"],
];

export function MyTeam({
  pool,
  team,
  onTeam,
  onOpen,
}: {
  pool: Player[];
  team: ImportedTeam | null;
  onTeam: (team: ImportedTeam | null) => void;
  onOpen: (player: Player) => void;
}) {
  const [entryId, setEntryId] = useState(() => {
    try {
      return typeof window === "undefined" ? "" : window.localStorage.getItem("fpl-lens-entry") ?? "";
    } catch {
      return "";
    }
  });
  const [state, setState] = useState<"idle" | "loading" | "error">("idle");
  const [error, setError] = useState("");
  const [sellId, setSellId] = useState<number | null>(null);

  const load = async (event?: FormEvent) => {
    event?.preventDefault();
    const id = entryId.trim();
    if (!/^\d+$/.test(id)) {
      setState("error");
      setError("Enter the number from your FPL team page URL (fantasy.premierleague.com/entry/NUMBER/...).");
      return;
    }
    setState("loading");
    setError("");
    try {
      const response = await fetch(`/api/team/${id}`);
      const payload = (await response.json()) as ImportedTeam & { error?: string };
      if (!response.ok) throw new Error(payload.error ?? "Could not load that team.");
      onTeam(payload);
      setState("idle");
      try {
        window.localStorage.setItem("fpl-lens-entry", id);
      } catch {
        // Storage can be unavailable; the import still works.
      }
    } catch (caught) {
      setState("error");
      setError(caught instanceof Error ? caught.message : "Could not load that team.");
    }
  };

  // Use the freshest projection for each owned player.
  const live = useMemo(() => new Map(pool.map((player) => [player.id, player])), [pool]);
  const owned = useMemo(
    () => (team?.owned ?? []).map((pick) => ({ ...pick, player: live.get(pick.element) ?? pick.player })),
    [team, live],
  );
  const lineup = useMemo(() => bestLineup(owned.map((pick) => pick.player)), [owned]);
  const problems = owned.filter((pick) => availabilityIssue(pick.player));
  const gw = results.headline.gameweek;
  const half = gw <= 19 ? 1 : 2;

  const replacements = useMemo(() => {
    if (!team || sellId === null) return [];
    const sell = owned.find((pick) => pick.element === sellId);
    if (!sell) return [];
    const budget = sell.selling_price / 10 + team.manager.bank;
    const ownedIds = new Set(owned.map((pick) => pick.element));
    const clubCounts = owned.reduce<Record<string, number>>((counts, pick) => {
      counts[pick.player.team] = (counts[pick.player.team] ?? 0) + 1;
      return counts;
    }, {});
    return pool
      .filter(
        (player) =>
          player.position === sell.player.position &&
          !ownedIds.has(player.id) &&
          player.price <= budget + 1e-9 &&
          (player.team === sell.player.team || (clubCounts[player.team] ?? 0) < 3),
      )
      .map((player) => ({
        player,
        gain6: player.sixWeekProjected - sell.player.sixWeekProjected,
        gain1: player.projected - sell.player.projected,
        left: budget - player.price,
      }))
      .sort((a, b) => b.gain6 - a.gain6)
      .slice(0, 10);
  }, [team, sellId, owned, pool]);

  return (
    <div className="view">
      <Card title="Import your FPL team">
        <form className="import-form" onSubmit={load}>
          <input
            inputMode="numeric"
            value={entryId}
            onChange={(event) => setEntryId(event.target.value)}
            placeholder="Team ID, e.g. 123456"
            aria-label="FPL team ID"
          />
          <button type="submit" className="primary" disabled={state === "loading"}>
            {state === "loading" ? "Loading…" : team ? "Refresh" : "Import"}
          </button>
          {team && <button type="button" className="text-button" onClick={() => onTeam(null)}>Clear</button>}
        </form>
        {state === "error" && <p className="error" role="alert">{error}</p>}
        <p className="note">
          Your team ID is the number in the address of your FPL points page. Only public data is read; no login is needed.
          FPL hides this week&apos;s transfers until the deadline, so the squad shown is the one from GW{team?.picksEvent ?? "…"}.
        </p>
      </Card>

      {!team ? (
        <Empty>Import your team to see your best XI, captain, problem players and transfer options.</Empty>
      ) : (
        <>
          <div className="summary-strip">
            <Stat label={team.manager.teamName} value={team.manager.playerName} />
            <Stat label="Overall rank" value={team.manager.overallRank.toLocaleString()} sub={`top ${team.manager.percentile.toFixed(1)}%`} />
            <Stat label="Total points" value={team.manager.points} />
            <Stat label="Team value" value={money(team.manager.squadValue)} sub={`${money(team.manager.bank)} in the bank`} />
            <Stat label="Free transfers" value={team.transferPolicy?.freeTransfers ?? "—"} sub="estimated" />
            <Stat label={`GW${gw} projection`} value={one(lineup?.total ?? 0)} sub={`model's team: ${one(team.forecast.modelProjection)}`} />
          </div>

          <div className="grid-2">
            <Card title="Season so far">
              {team.history?.length ? <SeasonChart history={team.history} /> : <Empty>No gameweeks played yet.</Empty>}
            </Card>
            <Card title={`Chips (${half === 1 ? "first" : "second"} half of the season)`}>
              <ul className="chip-status">
                {CHIPS.map(([key, label]) => {
                  const used = (team.chipsUsed ?? []).find(
                    (chip) => chip.name === key && (half === 1 ? chip.event <= 19 : chip.event >= 20),
                  );
                  return (
                    <li key={key} className={used ? "used" : "available"}>
                      <span>{label}</span>
                      <strong>{used ? `Used GW${used.event}` : `Available until GW${half === 1 ? 19 : 38}`}</strong>
                    </li>
                  );
                })}
              </ul>
            </Card>
          </div>

          {(problems.length > 0 || (team.unavailable?.length ?? 0) > 0) && (
            <Card title="Players to sort out">
              <ul className="news-list">
                {(team.unavailable ?? []).map((item) => (
                  <li key={item.element}>
                    <strong>{item.name}</strong>
                    <span>{item.team} · {item.position}</span>
                    <span className="muted">Not expected to play (injured, suspended or left the league)</span>
                  </li>
                ))}
                {problems.map((pick) => (
                  <li key={pick.element}>
                    <button type="button" className="player-link" onClick={() => onOpen(pick.player)}>{pick.player.name}</button>
                    <span>{pick.player.team} · {pick.player.position}</span>
                    <span className="muted">{availabilityIssue(pick.player)}</span>
                  </li>
                ))}
              </ul>
            </Card>
          )}

          {lineup && (
            <Card title={`Best XI from your squad for GW${gw}`} aside={<span className="muted">{lineup.formation} · captain {lineup.captain?.name} · vice {lineup.vice?.name}</span>}>
              <Pitch xi={lineup.xi} bench={lineup.bench} captainId={lineup.captain?.id} viceId={lineup.vice?.id} onOpen={onOpen} />
            </Card>
          )}

          <Card title="Suggested transfers">
            {team.suggestions.length ? (
              <ul className="move-list">
                {team.suggestions.map((move) => (
                  <li key={`${move.sell.id}-${move.buy.id}`} className="move">
                    <div className="move-side out"><span className="move-tag">Out</span><strong>{move.sell.name}</strong><span className="muted small">{move.sell.team}</span></div>
                    <span className="move-arrow" aria-hidden>→</span>
                    <div className="move-side in">
                      <span className="move-tag">In</span>
                      <button type="button" className="player-link" onClick={() => onOpen(move.buy)}>{move.buy.name}</button>
                      <span className="muted small">{move.buy.team} · {money(move.buy.price)}</span>
                    </div>
                    <span className="gain">{signed(move.horizonGain)}</span>
                  </li>
                ))}
              </ul>
            ) : (
              <Empty>No transfer gains more than {one(team.transferPolicy?.hurdle ?? 4)} points over six gameweeks. Rolling the transfer is the better play.</Empty>
            )}
          </Card>

          <Card title="Find a replacement">
            <div className="filters">
              <select value={sellId ?? ""} onChange={(event) => setSellId(event.target.value ? Number(event.target.value) : null)} aria-label="Player to sell">
                <option value="">Choose a player to sell…</option>
                {owned
                  .slice()
                  .sort((a, b) => a.player.sixWeekProjected - b.player.sixWeekProjected)
                  .map((pick) => (
                    <option key={pick.element} value={pick.element}>
                      {pick.player.name} ({pick.player.position}, sells for {money(pick.selling_price / 10)}, {one(pick.player.sixWeekProjected)} pts / 6 GW)
                    </option>
                  ))}
              </select>
            </div>
            {sellId === null ? (
              <Empty>Pick a player to see the best affordable replacements in the same position.</Empty>
            ) : replacements.length ? (
              <div className="table-scroll">
                <table className="data-table">
                  <thead>
                    <tr><th>Player</th><th>Pos</th><th>Next</th><th className="num">Price</th><th className="num">Gain next GW</th><th className="num">Gain 6 GW</th><th className="num">Bank after</th></tr>
                  </thead>
                  <tbody>
                    {replacements.map(({ player, gain1, gain6, left }) => (
                      <tr key={player.id}>
                        <td><button type="button" className="player-link" onClick={() => onOpen(player)}>{player.name}</button><Flag player={player} /><span className="muted small"> {player.team}</span></td>
                        <td><PositionTag position={player.position} /></td>
                        <td>{fixtureLabel(player)}</td>
                        <td className="num">{one(player.price)}</td>
                        <td className={`num ${gain1 >= 0 ? "pos-val" : "neg-val"}`}>{signed(gain1)}</td>
                        <td className={`num strong ${gain6 >= 0 ? "pos-val" : "neg-val"}`}>{signed(gain6)}</td>
                        <td className="num">{money(left)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : (
              <Empty>No affordable replacement in that position.</Empty>
            )}
          </Card>
        </>
      )}
    </div>
  );
}

function SeasonChart({ history }: { history: NonNullable<ImportedTeam["history"]> }) {
  const maxPoints = Math.max(1, ...history.map((row) => row.points));
  return (
    <div className="season-bars" role="img" aria-label="Points per gameweek">
      {history.map((row) => (
        <div key={row.event} className="season-bar" title={`GW${row.event}: ${row.points} pts, rank ${row.overallRank.toLocaleString()}`}>
          <span className="season-bar-value">{row.points}</span>
          <i style={{ height: `${(row.points / maxPoints) * 100}%` }} />
          <span className="season-bar-label">{row.event}</span>
        </div>
      ))}
    </div>
  );
}

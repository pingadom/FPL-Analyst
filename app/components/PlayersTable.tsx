import { useMemo, useState } from "react";
import { fixtureLabel, one } from "./format";
import type { Player, Position } from "./types";
import { POSITIONS } from "./types";
import { Card, Empty, Flag, PositionTag } from "./ui";

type SortKey =
  | "name" | "price" | "ownership" | "projected" | "sixWeek" | "value"
  | "start" | "return5" | "haul8" | "rise" | "fall";

const COLUMNS: Array<{ key: SortKey; label: string; title: string; get: (p: Player) => number | string; render?: (p: Player) => React.ReactNode }> = [
  { key: "price", label: "£", title: "Current price (£m)", get: (p) => p.price, render: (p) => one(p.price) },
  { key: "ownership", label: "Own", title: "Selected by % of managers", get: (p) => p.ownership, render: (p) => `${one(p.ownership)}%` },
  { key: "projected", label: "xPts", title: "Expected points next gameweek", get: (p) => p.projected, render: (p) => one(p.projected) },
  { key: "sixWeek", label: "6 GW", title: "Expected points over the next six gameweeks", get: (p) => p.sixWeekProjected, render: (p) => one(p.sixWeekProjected) },
  { key: "value", label: "Pts/£", title: "Six-gameweek expected points per £1m", get: (p) => p.sixWeekProjected / Math.max(p.price, 0.1), render: (p) => (p.sixWeekProjected / Math.max(p.price, 0.1)).toFixed(2) },
  { key: "start", label: "Start", title: "Probability of starting", get: (p) => p.minutesModel.startProbability, render: (p) => `${p.minutesModel.startProbability}%` },
  { key: "return5", label: "5+", title: "Probability of 5 or more points", get: (p) => p.distribution.return5Probability, render: (p) => `${p.distribution.return5Probability}%` },
  { key: "haul8", label: "8+", title: "Probability of 8 or more points", get: (p) => p.distribution.haul8Probability, render: (p) => `${p.distribution.haul8Probability}%` },
  { key: "rise", label: "▲", title: "Chance of a price rise", get: (p) => p.marketForecast.priceRiseProbability, render: (p) => (p.marketForecast.priceRiseProbability >= 20 ? `${p.marketForecast.priceRiseProbability}%` : "") },
  { key: "fall", label: "▼", title: "Chance of a price fall", get: (p) => p.marketForecast.priceFallProbability, render: (p) => (p.marketForecast.priceFallProbability >= 20 ? `${p.marketForecast.priceFallProbability}%` : "") },
];

const PRESETS: Array<{ label: string; apply: () => Partial<Filters> }> = [
  { label: "Best next GW", apply: () => ({ sort: "projected", desc: true }) },
  { label: "Best over 6 GWs", apply: () => ({ sort: "sixWeek", desc: true }) },
  { label: "Best value", apply: () => ({ sort: "value", desc: true, startersOnly: true }) },
  { label: "Differentials", apply: () => ({ sort: "sixWeek", desc: true, maxOwnership: 10 }) },
  { label: "Price risers", apply: () => ({ sort: "rise", desc: true }) },
  { label: "Price fallers", apply: () => ({ sort: "fall", desc: true }) },
];

type Filters = {
  query: string;
  position: Position | "ALL";
  team: string;
  maxPrice: number;
  maxOwnership: number;
  startersOnly: boolean;
  hideFlagged: boolean;
  sort: SortKey;
  desc: boolean;
};

const DEFAULTS: Filters = {
  query: "",
  position: "ALL",
  team: "ALL",
  maxPrice: 16,
  maxOwnership: 100,
  startersOnly: false,
  hideFlagged: false,
  sort: "sixWeek",
  desc: true,
};

export function PlayersTable({
  pool,
  onOpen,
  compareIds,
  onToggleCompare,
}: {
  pool: Player[];
  onOpen: (player: Player) => void;
  compareIds: number[];
  onToggleCompare: (player: Player) => void;
}) {
  const [filters, setFilters] = useState<Filters>(DEFAULTS);
  const [limit, setLimit] = useState(60);
  const set = (patch: Partial<Filters>) => {
    setFilters((current) => ({ ...current, ...patch }));
    setLimit(60);
  };
  const teams = useMemo(() => [...new Set(pool.map((player) => player.team))].sort(), [pool]);
  const rows = useMemo(() => {
    const query = filters.query.trim().toLowerCase();
    const column = COLUMNS.find((item) => item.key === filters.sort);
    const filtered = pool.filter(
      (player) =>
        (!query || player.name.toLowerCase().includes(query) || player.team.toLowerCase() === query) &&
        (filters.position === "ALL" || player.position === filters.position) &&
        (filters.team === "ALL" || player.team === filters.team) &&
        player.price <= filters.maxPrice + 1e-9 &&
        player.ownership <= filters.maxOwnership &&
        (!filters.startersOnly || player.minutesModel.startProbability >= 70) &&
        (!filters.hideFlagged ||
          (player.minutesModel.availabilityEvidence.status === "a" &&
            player.minutesModel.availabilityEvidence.chance >= 100)),
    );
    const value = (player: Player) => (filters.sort === "name" ? player.name : column?.get(player) ?? 0);
    return filtered.sort((a, b) => {
      const left = value(a);
      const right = value(b);
      const order = typeof left === "string" ? left.localeCompare(String(right)) : left - Number(right);
      return filters.desc ? -order : order;
    });
  }, [pool, filters]);

  const sortBy = (key: SortKey) =>
    set({ sort: key, desc: filters.sort === key ? !filters.desc : key !== "name" });
  const arrow = (key: SortKey) => (filters.sort === key ? (filters.desc ? " ↓" : " ↑") : "");

  return (
    <div className="view">
      <Card>
        <div className="filters">
          <input
            className="search"
            type="search"
            placeholder="Search player or team (e.g. ARS)"
            value={filters.query}
            onChange={(event) => set({ query: event.target.value })}
            aria-label="Search players"
          />
          <div className="segmented" role="group" aria-label="Position">
            {(["ALL", ...POSITIONS] as const).map((position) => (
              <button
                key={position}
                type="button"
                className={filters.position === position ? "active" : ""}
                aria-pressed={filters.position === position}
                onClick={() => set({ position })}
              >
                {position === "ALL" ? "All" : position}
              </button>
            ))}
          </div>
          <select value={filters.team} onChange={(event) => set({ team: event.target.value })} aria-label="Team">
            <option value="ALL">All teams</option>
            {teams.map((team) => <option key={team} value={team}>{team}</option>)}
          </select>
          <label className="range-filter">
            Max £{one(filters.maxPrice)}m
            <input type="range" min={3.5} max={16} step={0.5} value={filters.maxPrice} onChange={(event) => set({ maxPrice: Number(event.target.value) })} />
          </label>
          <label className="check"><input type="checkbox" checked={filters.startersOnly} onChange={(event) => set({ startersOnly: event.target.checked })} /> Likely starters (70%+)</label>
          <label className="check"><input type="checkbox" checked={filters.hideFlagged} onChange={(event) => set({ hideFlagged: event.target.checked })} /> Hide flagged</label>
          {filters.maxOwnership < 100 && (
            <button type="button" className="chip-button" onClick={() => set({ maxOwnership: 100 })}>Owned ≤ {filters.maxOwnership}% ✕</button>
          )}
          <button type="button" className="text-button" onClick={() => set(DEFAULTS)}>Reset</button>
        </div>
        <div className="presets">
          {PRESETS.map((preset) => (
            <button key={preset.label} type="button" className="chip-button" onClick={() => set({ ...DEFAULTS, ...preset.apply() })}>
              {preset.label}
            </button>
          ))}
          <span className="muted small">{rows.length} players · tick up to 4 to compare</span>
        </div>
      </Card>

      <Card>
        {rows.length ? (
          <div className="table-scroll">
            <table className="data-table players">
              <thead>
                <tr>
                  <th aria-label="Compare" />
                  <th><button type="button" className="sort" onClick={() => sortBy("name")}>Player{arrow("name")}</button></th>
                  <th>Pos</th>
                  <th>Next</th>
                  {COLUMNS.map((column) => (
                    <th key={column.key} className="num" title={column.title}>
                      <button type="button" className="sort" onClick={() => sortBy(column.key)}>{column.label}{arrow(column.key)}</button>
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {rows.slice(0, limit).map((player) => (
                  <tr key={player.id} className={compareIds.includes(player.id) ? "selected" : ""}>
                    <td>
                      <input
                        type="checkbox"
                        aria-label={`Compare ${player.name}`}
                        checked={compareIds.includes(player.id)}
                        disabled={!compareIds.includes(player.id) && compareIds.length >= 4}
                        onChange={() => onToggleCompare(player)}
                      />
                    </td>
                    <td className="name-cell">
                      <button type="button" className="player-link" onClick={() => onOpen(player)}>{player.name}</button>
                      <Flag player={player} />
                      <span className="muted small"> {player.team}</span>
                    </td>
                    <td><PositionTag position={player.position} /></td>
                    <td className="small">{fixtureLabel(player)}</td>
                    {COLUMNS.map((column) => (
                      <td key={column.key} className={`num ${column.key === filters.sort ? "strong" : ""}`}>
                        {column.render ? column.render(player) : String(column.get(player))}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <Empty>No players match these filters.</Empty>
        )}
        {rows.length > limit && (
          <button type="button" className="more-button" onClick={() => setLimit((value) => value + 60)}>
            Show more ({rows.length - limit} left)
          </button>
        )}
      </Card>
    </div>
  );
}

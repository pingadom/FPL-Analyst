import { useEffect, useState } from "react";
import { fixtureLabel, money, one } from "./format";
import type { Player } from "./types";
import { PositionTag, Stat } from "./ui";

const COMPONENTS: Array<[keyof Player["components"], string]> = [
  ["appearance", "Playing time"],
  ["goals", "Goals"],
  ["assists", "Assists"],
  ["cleanSheet", "Clean sheet"],
  ["defence", "Defensive contributions"],
  ["bonus", "Bonus"],
  ["adjustment", "Other adjustments"],
];

export function PlayerDetail({
  player,
  onClose,
  inCompare,
  onToggleCompare,
}: {
  player: Player;
  onClose: () => void;
  inCompare: boolean;
  onToggleCompare: (player: Player) => void;
}) {
  useEffect(() => {
    const onKey = (event: KeyboardEvent) => event.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);
  const evidence = player.minutesModel.availabilityEvidence;
  const maxComponent = Math.max(0.5, ...Object.values(player.components).map(Math.abs));
  return (
    <div className="drawer-backdrop">
      <button type="button" className="drawer-dismiss" aria-label="Close player details" onClick={onClose} />
      <aside className="drawer" role="dialog" aria-modal="true" aria-label={player.name}>
        <header className="drawer-head">
          <div>
            <div className="drawer-meta"><PositionTag position={player.position} /> {player.team} · {money(player.price)} · {one(player.ownership)}% owned</div>
            <h2>{player.name}</h2>
            <div className="muted">Next: {fixtureLabel(player)} · {player.archetype}</div>
          </div>
          <div className="drawer-actions">
            <button type="button" className="chip-button" onClick={() => onToggleCompare(player)}>
              {inCompare ? "Remove from compare" : "Add to compare"}
            </button>
            <button type="button" className="close" onClick={onClose} aria-label="Close">✕</button>
          </div>
        </header>

        {(evidence.status !== "a" || evidence.chance < 100) && (
          <div className="alert">
            <strong>{evidence.chance}% chance of playing.</strong> {evidence.officialNews}
          </div>
        )}

        <div className="stat-row">
          <Stat label="xPts next GW" value={one(player.projected)} />
          <Stat label="xPts next 6 GW" value={one(player.sixWeekProjected)} />
          <Stat label="Expected minutes" value={player.expectedMinutes} sub={`${player.minutesModel.startProbability}% start · ${player.minutesModel.sixtyProbability}% 60+`} />
          <Stat label="Pts per £m (6 GW)" value={(player.sixWeekProjected / Math.max(player.price, 0.1)).toFixed(2)} />
        </div>

        <section className="drawer-section">
          <h3>Where the points come from (next GW)</h3>
          <div className="bars">
            {COMPONENTS.map(([key, label]) => {
              const value = player.components[key];
              if (Math.abs(value) < 0.005) return null;
              return (
                <div className="bar-row" key={key}>
                  <span>{label}</span>
                  <i className={value < 0 ? "neg" : ""}><b style={{ width: `${(Math.abs(value) / maxComponent) * 100}%` }} /></i>
                  <strong>{value.toFixed(2)}</strong>
                </div>
              );
            })}
          </div>
        </section>

        <section className="drawer-section">
          <h3>Chance of each kind of week</h3>
          <div className="stat-row small">
            <Stat label="Blank (2 or fewer)" value={`${player.distribution.blankProbability}%`} />
            <Stat label="Return (5+)" value={`${player.distribution.return5Probability}%`} />
            <Stat label="Haul (8+)" value={`${player.distribution.haul8Probability}%`} />
          </div>
        </section>

        <section className="drawer-section two-col">
          <div>
            <h3>Track record</h3>
            <dl className="kv">
              <dt>Appearances</dt><dd>{player.history.matches}</dd>
              <dt>Points per appearance</dt><dd>{player.history.average.toFixed(2)}</dd>
              <dt>Points per 90</dt><dd>{player.history.per90.toFixed(2)}</dd>
              <dt>Returned 5+ pts</dt><dd>{player.history.returnRate}%</dd>
              <dt>Vs {player.opponent}</dt>
              <dd>{player.opponentHistory.matches ? `${player.opponentHistory.average.toFixed(1)} pts in ${player.opponentHistory.matches}` : "No previous meetings"}</dd>
            </dl>
          </div>
          <div>
            <h3>Team and fixture</h3>
            <dl className="kv">
              <dt>{player.team} attack / defence</dt><dd>#{player.teamContext.attackRank} / #{player.teamContext.defenceRank} of 20</dd>
              <dt>Expected goals for / against</dt><dd>{player.teamContext.expectedGoalsFor.toFixed(2)} / {player.teamContext.expectedGoalsAgainst.toFixed(2)}</dd>
              <dt>Clean-sheet chance</dt><dd>{player.teamContext.cleanSheetProbability}% (6 GW avg {player.teamContext.horizonCleanSheetProbability}%)</dd>
              <dt>Win chance (model / market)</dt><dd>{player.teamContext.modelWinProbability}% / {player.teamContext.marketWinProbability}%</dd>
            </dl>
          </div>
        </section>

        <section className="drawer-section two-col">
          <div>
            <h3>Role</h3>
            <dl className="kv">
              <dt>Set pieces</dt><dd>{player.setPieces.length ? player.setPieces.join(", ") : "None"}</dd>
              {player.position !== "GK" && <><dt>Defensive actions / 90</dt><dd>{player.defenderModel.actionRate}</dd></>}
              <dt>Bonus points system / match</dt><dd>{player.defenderModel.bpsRate}</dd>
              <dt>Notes</dt><dd>{player.riskFlags.filter((flag) => flag !== "No major flag").join(", ") || "—"}</dd>
            </dl>
          </div>
          <div>
            <h3>Price and ownership</h3>
            <dl className="kv">
              <dt>Price rise tonight</dt><dd>{player.marketForecast.priceRiseProbability}%</dd>
              <dt>Price fall tonight</dt><dd>{player.marketForecast.priceFallProbability}%</dd>
              <dt>Most-owned alternative in his position</dt>
              <dd>{player.comparison.popularRival} ({one(player.comparison.popularRivalOwnership)}%, {one(player.comparison.popularRivalProjection)} xPts)</dd>
              <dt>Rank in position</dt><dd>#{player.comparison.positionRank} of {player.comparison.positionPlayers}</dd>
            </dl>
          </div>
        </section>
      </aside>
    </div>
  );
}

const COMPARE_ROWS: Array<{ label: string; get: (p: Player) => number; format?: (v: number) => string; lowerBetter?: boolean }> = [
  { label: "Price", get: (p) => p.price, format: (v) => money(v), lowerBetter: true },
  { label: "xPts next GW", get: (p) => p.projected, format: one },
  { label: "xPts next 6 GW", get: (p) => p.sixWeekProjected, format: one },
  { label: "Pts per £m (6 GW)", get: (p) => p.sixWeekProjected / Math.max(p.price, 0.1), format: (v) => v.toFixed(2) },
  { label: "Start probability", get: (p) => p.minutesModel.startProbability, format: (v) => `${v}%` },
  { label: "Expected minutes", get: (p) => p.expectedMinutes, format: (v) => String(v) },
  { label: "5+ points chance", get: (p) => p.distribution.return5Probability, format: (v) => `${v}%` },
  { label: "8+ points chance", get: (p) => p.distribution.haul8Probability, format: (v) => `${v}%` },
  { label: "Points per 90 (history)", get: (p) => p.history.per90, format: (v) => v.toFixed(2) },
  { label: "Ownership", get: (p) => p.ownership, format: (v) => `${one(v)}%`, lowerBetter: true },
];

export function ComparePanel({
  players,
  onRemove,
  onClear,
  onOpen,
}: {
  players: Player[];
  onRemove: (player: Player) => void;
  onClear: () => void;
  onOpen: (player: Player) => void;
}) {
  const [expanded, setExpanded] = useState(false);
  if (!players.length) return null;
  return (
    <section className={`compare ${expanded ? "expanded" : ""}`} aria-label="Player comparison">
      <header className="compare-head">
        <button type="button" className="compare-toggle" aria-expanded={expanded} onClick={() => setExpanded((value) => !value)}>
          <strong>Compare ({players.length})</strong>
          <span className="muted">{players.map((player) => player.name).join(", ")}</span>
          <span className="compare-chevron" aria-hidden>{expanded ? "▾" : "▴"}</span>
        </button>
        <button type="button" className="text-button" onClick={onClear}>Clear</button>
      </header>
      {expanded && <div className="table-scroll">
        <table className="data-table compare-table">
          <thead>
            <tr>
              <th />
              {players.map((player) => (
                <th key={player.id}>
                  <button type="button" className="player-link" onClick={() => onOpen(player)}>{player.name}</button>
                  <button type="button" className="remove" onClick={() => onRemove(player)} aria-label={`Remove ${player.name}`}>✕</button>
                  <div className="muted small">{player.team} · {fixtureLabel(player)}</div>
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {COMPARE_ROWS.map((row) => {
              const values = players.map(row.get);
              const best = row.lowerBetter ? Math.min(...values) : Math.max(...values);
              return (
                <tr key={row.label}>
                  <th scope="row">{row.label}</th>
                  {values.map((value, index) => (
                    <td key={players[index].id} className={players.length > 1 && value === best ? "best" : ""}>
                      {row.format ? row.format(value) : value}
                    </td>
                  ))}
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>}
    </section>
  );
}

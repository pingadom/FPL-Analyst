import { useMemo, useState } from "react";
import { buildOptimizedSquad } from "../lib/squad-optimizer.mjs";
import { money, one, positionOrder } from "./format";
import { Pitch, SquadTable } from "./Pitch";
import type { ImportedTeam, Player } from "./types";
import { Card, Empty, Stat } from "./ui";

type Mode = "wildcard" | "freehit";
type Built = { squad: Player[]; xi: Player[]; captain: Player; bench: Player[]; spend: number };

function PlayerPicker({
  pool,
  label,
  chosen,
  onAdd,
  onRemove,
}: {
  pool: Player[];
  label: string;
  chosen: Player[];
  onAdd: (player: Player) => void;
  onRemove: (player: Player) => void;
}) {
  const [text, setText] = useState("");
  const listId = `${label.replace(/\W/g, "")}-list`;
  const key = (player: Player) => `${player.name} (${player.team}, ${player.position}, £${player.price.toFixed(1)})`;
  const byKey = useMemo(() => new Map(pool.map((player) => [key(player), player])), [pool]);
  return (
    <div className="picker">
      <label>
        <span className="stat-label">{label}</span>
        <input
          list={listId}
          value={text}
          placeholder="Type a name…"
          onChange={(event) => {
            const player = byKey.get(event.target.value);
            if (player) {
              onAdd(player);
              setText("");
            } else {
              setText(event.target.value);
            }
          }}
        />
      </label>
      <datalist id={listId}>
        {pool.map((player) => <option key={player.id} value={key(player)} />)}
      </datalist>
      <div className="tags">
        {chosen.map((player) => (
          <button key={player.id} type="button" className="tag" onClick={() => onRemove(player)} aria-label={`Remove ${player.name}`}>
            {player.name} ✕
          </button>
        ))}
      </div>
    </div>
  );
}

export function Planner({
  pool,
  team,
  onOpen,
}: {
  pool: Player[];
  team: ImportedTeam | null;
  onOpen: (player: Player) => void;
}) {
  const [mode, setMode] = useState<Mode>("wildcard");
  const [budget, setBudget] = useState(() =>
    team ? Number((team.manager.squadValue + team.manager.bank).toFixed(1)) : 100,
  );
  const [locked, setLocked] = useState<Player[]>([]);
  const [excluded, setExcluded] = useState<Player[]>([]);
  const [built, setBuilt] = useState<Built | null>(null);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  const build = () => {
    setBusy(true);
    setError("");
    // Let the button repaint before the solver blocks the thread.
    window.setTimeout(() => {
      try {
        const scored = pool.map((player) => {
          const value = mode === "freehit" ? player.projected : player.sixWeekProjected / 6;
          return {
            ...player,
            liveScore: player.strategyScores.balanced,
            optimizerUtility: value,
            optimization: { ...player.optimization, captainUtility: value },
          };
        });
        const result = buildOptimizedSquad(scored, {
          budget,
          lockedIds: locked.map((player) => player.id),
          excludedIds: excluded.map((player) => player.id),
          allowStrategicBank: true,
        }) as Built | null;
        if (!result) throw new Error("No squad found.");
        setBuilt(result);
      } catch {
        setBuilt(null);
        setError(
          "No legal squad fits those settings. Check the budget, and that locked players respect 2 GK / 5 DEF / 5 MID / 3 FWD and at most 3 per club.",
        );
      } finally {
        setBusy(false);
      }
    }, 20);
  };

  const live = new Map(pool.map((player) => [player.id, player]));
  const squad = built?.squad.map((player) => live.get(player.id) ?? player) ?? [];
  const xi = built?.xi.map((player) => live.get(player.id) ?? player) ?? [];
  // Whatever the planning horizon, the armband this week goes to this week's best.
  const [captain, vice] = [...xi].sort((a, b) => b.projected - a.projected);
  const xiProjection = xi.reduce((sum, player) => sum + player.projected, 0) + (captain?.projected ?? 0);
  const sixWeek = xi.reduce((sum, player) => sum + player.sixWeekProjected, 0);

  return (
    <div className="view">
      <Card title="Wildcard and Free Hit planner">
        <div className="planner-controls">
          <div className="segmented" role="group" aria-label="Chip">
            <button type="button" className={mode === "wildcard" ? "active" : ""} aria-pressed={mode === "wildcard"} onClick={() => setMode("wildcard")}>Wildcard · best over 6 GWs</button>
            <button type="button" className={mode === "freehit" ? "active" : ""} aria-pressed={mode === "freehit"} onClick={() => setMode("freehit")}>Free Hit · best this GW</button>
          </div>
          <label className="budget">
            <span className="stat-label">Budget (£m)</span>
            <input type="number" min={80} max={120} step={0.1} value={budget} onChange={(event) => setBudget(Number(event.target.value))} />
          </label>
          <PlayerPicker pool={pool} label="Must include" chosen={locked} onAdd={(player) => setLocked((list) => list.some((p) => p.id === player.id) ? list : [...list, player])} onRemove={(player) => setLocked((list) => list.filter((p) => p.id !== player.id))} />
          <PlayerPicker pool={pool} label="Leave out" chosen={excluded} onAdd={(player) => setExcluded((list) => list.some((p) => p.id === player.id) ? list : [...list, player])} onRemove={(player) => setExcluded((list) => list.filter((p) => p.id !== player.id))} />
          <button type="button" className="primary" onClick={build} disabled={busy || !pool.length}>
            {busy ? "Building…" : "Build squad"}
          </button>
        </div>
        {error && <p className="error" role="alert">{error}</p>}
        <p className="note">
          Picks the 15 players, starting XI and captain that score the most expected points within your budget, with at most three
          players per club. The bench is kept cheap so the money goes into the XI.
        </p>
      </Card>

      {built ? (
        <>
          <div className="summary-strip">
            <Stat label="Next GW (with captain)" value={one(xiProjection)} />
            <Stat label="XI over 6 GWs" value={one(sixWeek)} />
            <Stat label="Cost" value={money(built.spend)} sub={`${money(budget - built.spend)} left`} />
            <Stat label="Captain this GW" value={captain?.name ?? "—"} sub={vice ? `vice ${vice.name}` : undefined} />
          </div>
          <Card title="Squad">
            <Pitch xi={xi} bench={built.bench.map((player) => live.get(player.id) ?? player)} captainId={captain?.id} viceId={vice?.id} onOpen={onOpen} />
          </Card>
          <Card>
            <SquadTable players={[...squad].sort((a, b) => positionOrder(a.position) - positionOrder(b.position) || b.projected - a.projected)} captainId={captain?.id} viceId={vice?.id} onOpen={onOpen} />
          </Card>
        </>
      ) : (
        !error && <Empty>Set a budget and any must-have players, then build.</Empty>
      )}
    </div>
  );
}

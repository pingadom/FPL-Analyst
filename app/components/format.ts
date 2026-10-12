import type { Player, Position } from "./types";

export const one = (value: number) => value.toFixed(1);
export const money = (value: number) => `£${value.toFixed(1)}m`;
export const pct = (value: number) => `${Math.round(value)}%`;
export const signed = (value: number, digits = 1) =>
  `${value >= 0 ? "+" : ""}${value.toFixed(digits)}`;

export function countdown(deadline: string, now: number) {
  const left = Math.max(0, new Date(deadline).getTime() - now);
  const days = Math.floor(left / 86_400_000);
  const hours = Math.floor((left % 86_400_000) / 3_600_000);
  const minutes = Math.floor((left % 3_600_000) / 60_000);
  if (left === 0) return "Deadline passed";
  return days > 0 ? `${days}d ${hours}h` : `${hours}h ${minutes}m`;
}

export function deadlineLabel(deadline: string) {
  return new Date(deadline).toLocaleString("en-GB", {
    weekday: "short",
    day: "numeric",
    month: "short",
    hour: "2-digit",
    minute: "2-digit",
  });
}

export const positionOrder = (position: Position) => ["GK", "DEF", "MID", "FWD"].indexOf(position);

export function fixtureLabel(player: Pick<Player, "opponent" | "venue">) {
  return `${player.opponent} (${player.venue})`;
}

/** Availability summary, or null when the player is fully fit and nailed. */
export function availabilityIssue(player: Player): string | null {
  const evidence = player.minutesModel.availabilityEvidence;
  if (evidence.status !== "a" || evidence.chance < 100) {
    const news = evidence.officialNews && evidence.officialNews !== "No official flag" ? evidence.officialNews : "Flagged";
    return `${evidence.chance}% · ${news}`;
  }
  if (player.minutesModel.startProbability < 60) {
    return `${player.minutesModel.startProbability}% to start`;
  }
  return null;
}

/** Best legal XI and captain from a set of players, by next-GW projection. */
export function bestLineup(players: Player[]) {
  const formations = [
    [3, 4, 3], [3, 5, 2], [4, 4, 2], [4, 3, 3], [4, 5, 1], [5, 4, 1], [5, 3, 2], [5, 2, 3],
  ];
  const sorted = (position: Position) =>
    players.filter((player) => player.position === position).sort((a, b) => b.projected - a.projected);
  let best: { xi: Player[]; formation: string; total: number } | null = null;
  for (const [defenders, midfielders, forwards] of formations) {
    const xi = [
      ...sorted("GK").slice(0, 1),
      ...sorted("DEF").slice(0, defenders),
      ...sorted("MID").slice(0, midfielders),
      ...sorted("FWD").slice(0, forwards),
    ];
    if (xi.length !== 11) continue;
    const total = xi.reduce((sum, player) => sum + player.projected, 0);
    if (!best || total > best.total) best = { xi, formation: `${defenders}-${midfielders}-${forwards}`, total };
  }
  if (!best) return null;
  const xiIds = new Set(best.xi.map((player) => player.id));
  const byProjection = [...best.xi].sort((a, b) => b.projected - a.projected);
  const benchOutfield = players
    .filter((player) => !xiIds.has(player.id) && player.position !== "GK")
    .sort((a, b) => b.projected - a.projected);
  const benchKeeper = players.filter((player) => !xiIds.has(player.id) && player.position === "GK");
  return {
    xi: best.xi,
    formation: best.formation,
    captain: byProjection[0],
    vice: byProjection[1],
    bench: [...benchKeeper, ...benchOutfield],
    total: best.total + (byProjection[0]?.projected ?? 0),
  };
}

export function difficultyClass(value: number) {
  return `fdr fdr-${Math.max(1, Math.min(5, Math.round(value)))}`;
}

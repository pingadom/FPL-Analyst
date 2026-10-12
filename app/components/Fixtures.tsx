import { useEffect, useMemo, useState } from "react";
import { difficultyClass } from "./format";
import type { FixtureCell, FixturesPayload } from "./types";
import { Card, Empty } from "./ui";

type Mode = "official" | "attack" | "defence";

const MODES: Array<{ key: Mode; label: string; hint: string }> = [
  { key: "official", label: "FPL difficulty", hint: "The official 1–5 fixture difficulty rating." },
  { key: "attack", label: "For attackers", hint: "How good the opponent's defence is, by the model's team ratings." },
  { key: "defence", label: "For defenders", hint: "How good the opponent's attack is, by the model's team ratings." },
];

// League rank 1 (best) to 20 mapped onto the familiar 1-5 scale, home a little easier.
function rankDifficulty(rank: number, home: boolean) {
  const base = 5 - Math.floor((Math.max(1, Math.min(20, rank)) - 1) / 4);
  return Math.max(1, Math.min(5, base - (home ? 0.3 : 0)));
}

export function Fixtures({ highlightTeams }: { highlightTeams: string[] }) {
  const [data, setData] = useState<FixturesPayload | null>(null);
  const [failed, setFailed] = useState(false);
  const [mode, setMode] = useState<Mode>("official");
  const [horizon, setHorizon] = useState(6);

  useEffect(() => {
    const controller = new AbortController();
    fetch("/api/fixtures", { signal: controller.signal })
      .then((response) => response.json() as Promise<FixturesPayload>)
      .then(setData)
      .catch((error: unknown) => {
        if (!(error instanceof DOMException && error.name === "AbortError")) setFailed(true);
      });
    return () => controller.abort();
  }, []);

  const difficulty = (cell: FixtureCell) => {
    if (mode === "official" || !data) return cell.difficulty;
    const rating = data.modelRatings[cell.opponent];
    if (!rating) return cell.difficulty;
    return rankDifficulty(mode === "attack" ? rating.defence : rating.attack, cell.home);
  };

  const events = (data?.events ?? []).slice(0, horizon);
  const rows = useMemo(() => {
    if (!data) return [];
    return data.teams
      .map((team) => {
        // A blank counts as the worst week; a double counts as a good one.
        const score = events.reduce((sum, event) => {
          const cells = team.fixtures[event] ?? [];
          if (!cells.length) return sum + 6;
          return sum + cells.reduce((total, cell) => total + difficulty(cell), 0) - (cells.length - 1) * 3.5;
        }, 0);
        return { team, score: score / Math.max(1, events.length) };
      })
      .sort((a, b) => a.score - b.score);
    // difficulty depends on mode and data only
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [data, mode, horizon]);

  return (
    <div className="view">
      <Card
        title="Fixture planner"
        aside={
          <div className="toolbar">
            <div className="segmented" role="group" aria-label="Difficulty view">
              {MODES.map((item) => (
                <button key={item.key} type="button" title={item.hint} className={mode === item.key ? "active" : ""} aria-pressed={mode === item.key} onClick={() => setMode(item.key)}>
                  {item.label}
                </button>
              ))}
            </div>
            <select value={horizon} onChange={(event) => setHorizon(Number(event.target.value))} aria-label="Gameweeks">
              {[3, 4, 5, 6].map((value) => <option key={value} value={value}>Next {value} GWs</option>)}
            </select>
          </div>
        }
      >
        {!data && !failed && <Empty>Loading fixtures…</Empty>}
        {(failed || data?.source === "unavailable") && (
          <Empty>The official fixture list could not be loaded right now. Try again shortly.</Empty>
        )}
        {data && data.teams.length > 0 && (
          <div className="table-scroll">
            <table className="fixture-grid">
              <thead>
                <tr>
                  <th>Team</th>
                  {events.map((event) => <th key={event}>GW{event}</th>)}
                  <th className="num" title="Average difficulty over the period (lower is easier)">Avg</th>
                </tr>
              </thead>
              <tbody>
                {rows.map(({ team, score }) => (
                  <tr key={team.id} className={highlightTeams.includes(team.short) ? "highlight" : ""}>
                    <th scope="row">{team.short}</th>
                    {events.map((event) => {
                      const cells = team.fixtures[event] ?? [];
                      if (!cells.length) return <td key={event} className="fdr fdr-blank">—</td>;
                      return (
                        <td key={event} className="fixture-cell">
                          {cells.map((cell, index) => (
                            <span key={index} className={difficultyClass(difficulty(cell))}>
                              {cell.home ? cell.opponent : cell.opponent.toLowerCase()}
                            </span>
                          ))}
                        </td>
                      );
                    })}
                    <td className="num strong">{score.toFixed(1)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
        <p className="note">
          Home fixtures in capitals, away in lower case. Blank gameweeks count as the hardest week and double gameweeks as easier.
          {highlightTeams.length > 0 && " Highlighted rows are teams you own players from."}
        </p>
      </Card>
    </div>
  );
}

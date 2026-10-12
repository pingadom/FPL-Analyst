import { useMemo, useState } from "react";
import { fixtureLabel, one } from "./format";
import type { Player } from "./types";
import { Card, Empty, Flag, PositionTag } from "./ui";

export function Captains({
  pool,
  ownedIds,
  onOpen,
}: {
  pool: Player[];
  ownedIds: number[];
  onOpen: (player: Player) => void;
}) {
  const [mine, setMine] = useState(false);
  const options = useMemo(() => {
    const owned = new Set(ownedIds);
    return pool
      .filter((player) => !mine || owned.has(player.id))
      .filter((player) => player.minutesModel.startProbability >= 50)
      .sort((a, b) => b.projected - a.projected)
      .slice(0, 15);
  }, [pool, ownedIds, mine]);
  const top = options[0]?.projected ?? 1;

  return (
    <div className="view">
      <Card
        title="Captain options"
        aside={
          <label className="check">
            <input type="checkbox" checked={mine} disabled={!ownedIds.length} onChange={(event) => setMine(event.target.checked)} />
            Only my team{!ownedIds.length && " (import it on My team)"}
          </label>
        }
      >
        {options.length ? (
          <div className="table-scroll">
            <table className="data-table">
              <thead>
                <tr>
                  <th>#</th><th>Player</th><th>Pos</th><th>Fixture</th>
                  <th className="num" title="Expected points; captaincy doubles this">xPts</th>
                  <th />
                  <th className="num" title="Chance of 8 or more points">8+</th>
                  <th className="num" title="Chance of 2 or fewer points">Blank</th>
                  <th className="num">Start</th>
                  <th className="num">Owned</th>
                </tr>
              </thead>
              <tbody>
                {options.map((player, index) => (
                  <tr key={player.id}>
                    <td className="muted">{index + 1}</td>
                    <td>
                      <button type="button" className="player-link" onClick={() => onOpen(player)}>{player.name}</button>
                      <Flag player={player} />
                      <span className="muted small"> {player.team}</span>
                    </td>
                    <td><PositionTag position={player.position} /></td>
                    <td>{fixtureLabel(player)}</td>
                    <td className="num strong">{one(player.projected)}</td>
                    <td className="bar-cell"><i><b style={{ width: `${(player.projected / top) * 100}%` }} /></i></td>
                    <td className="num">{player.distribution.haul8Probability}%</td>
                    <td className="num">{player.distribution.blankProbability}%</td>
                    <td className="num">{player.minutesModel.startProbability}%</td>
                    <td className="num">{one(player.ownership)}%</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <Empty>No captain options to show.</Empty>
        )}
        <p className="note">
          Ranked by expected points. Across ten past seasons, simply captaining the highest expected score beat
          captaining by form, ownership, haul chance or price.
        </p>
      </Card>
    </div>
  );
}

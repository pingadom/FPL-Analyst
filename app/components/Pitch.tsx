import { fixtureLabel, one } from "./format";
import type { Player } from "./types";
import { Flag, PositionTag } from "./ui";

/** XI laid out by line, bench underneath in autosub order. */
export function Pitch({
  xi,
  bench,
  captainId,
  viceId,
  onOpen,
}: {
  xi: Player[];
  bench: Player[];
  captainId?: number;
  viceId?: number;
  onOpen: (player: Player) => void;
}) {
  const lines = (["GK", "DEF", "MID", "FWD"] as const).map((position) =>
    xi.filter((player) => player.position === position),
  );
  const card = (player: Player, benchIndex?: number) => (
    <button type="button" className="pitch-player" key={player.id} onClick={() => onOpen(player)}>
      <span className="pitch-badges">
        {player.id === captainId && <b className="armband">C</b>}
        {player.id === viceId && <b className="armband vice">V</b>}
        {benchIndex !== undefined && <b className="bench-order">{benchIndex === 0 ? "GK" : benchIndex}</b>}
        <Flag player={player} />
      </span>
      <span className="pitch-name">{player.name}</span>
      <span className="pitch-fixture">{fixtureLabel(player)}</span>
      <span className="pitch-xp">{one(player.projected)}</span>
    </button>
  );
  return (
    <div className="pitch-wrap">
      <div className="pitch">
        {lines.map((line, index) => (
          <div className="pitch-line" key={index}>
            {line.map((player) => card(player))}
          </div>
        ))}
      </div>
      <div className="pitch-bench" aria-label="Bench in autosub order">
        <span className="bench-label">Bench</span>
        {bench.map((player, index) => card(player, player.position === "GK" ? 0 : index))}
      </div>
    </div>
  );
}

export function SquadTable({
  players,
  captainId,
  viceId,
  onOpen,
  extra,
}: {
  players: Player[];
  captainId?: number;
  viceId?: number;
  onOpen: (player: Player) => void;
  extra?: { header: string; cell: (player: Player) => React.ReactNode };
}) {
  return (
    <div className="table-scroll">
      <table className="data-table">
        <thead>
          <tr>
            <th>Player</th>
            <th>Pos</th>
            <th>Fixture</th>
            <th className="num">Price</th>
            <th className="num">xPts</th>
            <th className="num">6 GW</th>
            <th className="num">Start</th>
            <th>Status</th>
            {extra && <th>{extra.header}</th>}
          </tr>
        </thead>
        <tbody>
          {players.map((player) => (
            <tr key={player.id}>
              <td>
                <button type="button" className="player-link" onClick={() => onOpen(player)}>
                  {player.name}
                </button>
                {player.id === captainId && <b className="armband small">C</b>}
                {player.id === viceId && <b className="armband vice small">V</b>}
                <span className="muted"> {player.team}</span>
              </td>
              <td><PositionTag position={player.position} /></td>
              <td>{fixtureLabel(player)}</td>
              <td className="num">{one(player.price)}</td>
              <td className="num strong">{one(player.projected)}</td>
              <td className="num">{one(player.sixWeekProjected)}</td>
              <td className="num">{player.minutesModel.startProbability}%</td>
              <td className="status-cell">
                <Flag player={player} />
                <span className="muted small">
                  {player.minutesModel.availabilityEvidence.officialNews !== "No official flag"
                    ? player.minutesModel.availabilityEvidence.officialNews
                    : ""}
                </span>
              </td>
              {extra && <td>{extra.cell(player)}</td>}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

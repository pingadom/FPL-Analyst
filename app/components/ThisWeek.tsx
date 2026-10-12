import results from "../data/model-results.json";
import { deadlineLabel, money, one, positionOrder, signed } from "./format";
import { Pitch, SquadTable } from "./Pitch";
import type { Player } from "./types";
import { Card, Empty, Stat } from "./ui";

type Headline = {
  gameweek: number;
  deadline: string;
  projected: number;
  formation: string;
  bank?: number;
  freeTransfers?: number;
  freeTransfersNext?: number;
  transferHurdle?: number;
  chip?: string | null;
  chipSignals?: Record<string, { signal: number; threshold: number; structural: boolean }>;
  transfers?: Array<{ out: string; in: string; gain: number }>;
  scenario: { p10: number; p90: number; probability70: number };
};

export function ThisWeek({ pool, onOpen }: { pool: Player[]; onOpen: (player: Player) => void }) {
  const headline = results.headline as unknown as Headline;
  // Prefer live-overlaid versions of the squad players (fresh availability news).
  const live = new Map(pool.map((player) => [player.id, player]));
  const squad = (results.squad as unknown as Player[]).map((player) => ({
    ...(live.get(player.id) ?? player),
    starter: player.starter,
    captain: player.captain,
    vice: player.vice,
    purchasePrice: player.purchasePrice,
  }));
  const xi = squad.filter((player) => player.starter);
  const bench = squad
    .filter((player) => !player.starter)
    .sort((a, b) => (a.position === "GK" ? -1 : b.position === "GK" ? 1 : b.projected - a.projected));
  const captain = squad.find((player) => player.captain);
  const vice = squad.find((player) => player.vice);
  const byName = (name: string) => pool.find((player) => player.name === name);
  const transfers = headline.transfers ?? [];
  const signals = Object.entries(headline.chipSignals ?? {});
  const flagged = squad.filter(
    (player) =>
      player.minutesModel.availabilityEvidence.status !== "a" ||
      player.minutesModel.availabilityEvidence.chance < 100,
  );

  return (
    <div className="view">
      <div className="summary-strip">
        <Stat label={`GW${headline.gameweek} projection`} value={one(headline.projected)} sub={`80% of simulated weeks: ${Math.round(headline.scenario.p10)}–${Math.round(headline.scenario.p90)}`} />
        <Stat label="Deadline" value={deadlineLabel(headline.deadline)} />
        <Stat label="Transfers" value={transfers.length ? `${transfers.length} planned` : "Roll"} sub={`${headline.freeTransfers ?? 1} free · ${headline.freeTransfersNext ?? 1} next week`} />
        <Stat label="Bank" value={money((headline.bank ?? 0) / 10)} />
        <Stat label="Chip" value={headline.chip ?? "None"} />
      </div>

      <div className="grid-2">
        <Card title="Transfers this week">
          {transfers.length ? (
            <ul className="move-list">
              {transfers.map((move) => {
                const out = byName(move.out);
                const incoming = byName(move.in);
                return (
                  <li key={`${move.out}-${move.in}`} className="move">
                    <div className="move-side out">
                      <span className="move-tag">Out</span>
                      {out ? <button type="button" className="player-link" onClick={() => onOpen(out)}>{move.out}</button> : <strong>{move.out}</strong>}
                      {out && <span className="muted small">{out.team} · {money(out.price)} · {one(out.sixWeekProjected)} pts / 6 GW</span>}
                    </div>
                    <span className="move-arrow" aria-hidden>→</span>
                    <div className="move-side in">
                      <span className="move-tag">In</span>
                      {incoming ? <button type="button" className="player-link" onClick={() => onOpen(incoming)}>{move.in}</button> : <strong>{move.in}</strong>}
                      {incoming && <span className="muted small">{incoming.team} · {money(incoming.price)} · {one(incoming.sixWeekProjected)} pts / 6 GW</span>}
                    </div>
                    <span className="gain">{signed(move.gain)}</span>
                  </li>
                );
              })}
            </ul>
          ) : (
            <Empty>No move gains enough over six weeks to be worth a transfer. Save it for next week.</Empty>
          )}
          <p className="note">
            A transfer is only made if it adds at least {one(headline.transferHurdle ?? 4)} expected points over the next six gameweeks.
            Gains shown are six-gameweek expected points.
          </p>
        </Card>

        <Card title="Chips" aside={<span className={`pill ${headline.chip ? "pill-go" : ""}`}>{headline.chip ? `Play ${headline.chip}` : "Hold all chips"}</span>}>
          {signals.length ? (
            <table className="data-table compact">
              <thead>
                <tr><th>Chip</th><th className="num">Value now</th><th className="num">Needed</th><th>Trigger</th></tr>
              </thead>
              <tbody>
                {signals.map(([chip, value]) => (
                  <tr key={chip}>
                    <td>{chip}</td>
                    <td className="num">{one(value.signal)}</td>
                    <td className="num">{one(value.threshold)}</td>
                    <td>{value.structural ? "Blank/double this week" : "—"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          ) : (
            <Empty>No chip is available to play this week.</Empty>
          )}
          <p className="note">
            Value now is the expected-point gain from playing the chip this week. The bar it must clear falls as the chip&apos;s deadline approaches.
          </p>
        </Card>
      </div>

      {flagged.length > 0 && (
        <Card title="Injury and availability news in this squad">
          <ul className="news-list">
            {flagged.map((player) => (
              <li key={player.id}>
                <button type="button" className="player-link" onClick={() => onOpen(player)}>{player.name}</button>
                <span>{player.minutesModel.availabilityEvidence.chance}% chance of playing</span>
                <span className="muted">{player.minutesModel.availabilityEvidence.officialNews}</span>
              </li>
            ))}
          </ul>
        </Card>
      )}

      <Card
        title={`Team for GW${headline.gameweek}`}
        aside={<span className="muted">{headline.formation} · captain {captain?.name ?? "—"} · vice {vice?.name ?? "—"}</span>}
      >
        <Pitch xi={xi} bench={bench} captainId={captain?.id} viceId={vice?.id} onOpen={onOpen} />
      </Card>

      <Card title="Full squad">
        {squad.length ? (
          <SquadTable
            players={[...squad].sort((a, b) => positionOrder(a.position) - positionOrder(b.position) || b.projected - a.projected)}
            captainId={captain?.id}
            viceId={vice?.id}
            onOpen={onOpen}
          />
        ) : (
          <Empty>No squad in the current model output.</Empty>
        )}
      </Card>
    </div>
  );
}

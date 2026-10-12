import history from "../data/pick-history.json";
import results from "../data/model-results.json";
import { one, signed } from "./format";
import { Card, Empty, Stat } from "./ui";

// The feeds the numbers actually come from (the artifact also lists background reading).
const DATA_SOURCES = [
  { label: "Official FPL API", url: "https://fantasy.premierleague.com/api/bootstrap-static/" },
  { label: "historical FPL archive", url: "https://github.com/vaastav/Fantasy-Premier-League" },
  { label: "football-data.co.uk match odds", url: "https://www.football-data.co.uk/data.php" },
  { label: "Matchbook exchange prices", url: "https://api.matchbook.com/edge/rest/events?exchange-type=back-lay&sport-ids=15" },
  { label: "Opta match predictions", url: "https://theanalyst.com/articles/premier-league-match-predictions" },
];

type Entry = {
  season: string;
  gameweek: number;
  captain: string;
  projectedPoints: number;
  scored: boolean;
  chip?: string | null;
  transfers?: Array<{ out: string; in: string }>;
  realised?: { total: number; xiPoints: number; captainBonus: number; benchPoints: number; startersWhoDidNotPlay: number };
};

export function TrackRecord() {
  const entries = (history.entries as unknown as Entry[]).filter(
    (entry) => entry.season === results.headline.season,
  );
  const scored = entries.filter((entry) => entry.scored && entry.realised);
  const actual = scored.reduce((sum, entry) => sum + (entry.realised?.total ?? 0), 0);
  const projected = scored.reduce((sum, entry) => sum + entry.projectedPoints, 0);
  const seasons = results.backtest;
  const maxPoints = Math.max(...seasons.map((season) => Math.max(season.points, season.top500Target)));
  const minPoints = Math.min(...seasons.map((season) => Math.min(season.points, season.top500Target))) - 150;
  const average = seasons.reduce((sum, season) => sum + season.points, 0) / seasons.length;
  const targetAverage = seasons.reduce((sum, season) => sum + season.top500Target, 0) / seasons.length;

  return (
    <div className="view">
      <Card title={`Live picks, ${results.headline.season}`}>
        <p className="note">
          Every week the model&apos;s team is saved before the deadline and scored once the gameweek ends, so these are real
          out-of-sample results, not a replay.
        </p>
        {scored.length > 0 && (
          <div className="summary-strip">
            <Stat label="Gameweeks scored" value={scored.length} />
            <Stat label="Points" value={actual} sub={`${one(actual / scored.length)} a week`} />
            <Stat label="Projected" value={one(projected)} sub={`${signed(actual - projected)} vs projection`} />
          </div>
        )}
        {entries.length ? (
          <div className="table-scroll">
            <table className="data-table">
              <thead>
                <tr>
                  <th>GW</th><th>Captain</th><th>Chip</th><th>Transfers</th>
                  <th className="num">Projected</th><th className="num">Scored</th><th className="num">Captain</th><th className="num">Bench</th>
                </tr>
              </thead>
              <tbody>
                {[...entries].reverse().map((entry) => (
                  <tr key={entry.gameweek}>
                    <td>{entry.gameweek}</td>
                    <td>{entry.captain}</td>
                    <td>{entry.chip ?? "—"}</td>
                    <td className="small">{entry.transfers?.length ? entry.transfers.map((move) => `${move.out} → ${move.in}`).join(", ") : "—"}</td>
                    <td className="num">{one(entry.projectedPoints)}</td>
                    <td className="num strong">{entry.realised ? entry.realised.total : "pending"}</td>
                    <td className="num">{entry.realised ? entry.realised.captainBonus : ""}</td>
                    <td className="num muted">{entry.realised ? entry.realised.benchPoints : ""}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <Empty>No picks recorded for this season yet.</Empty>
        )}
        <p className="note">Scores are the starting XI plus the captain&apos;s extra points; automatic substitutions are not applied.</p>
      </Card>

      <Card title="Past seasons, replayed">
        <p className="note">
          The same model replayed through {seasons.length} past seasons: one squad, carried from week to week with real
          transfer, budget and chip rules, using only information available before each deadline. The grey line is an
          estimate of the score needed to finish in the top 500,000.
        </p>
        <div className="summary-strip">
          <Stat label="Average season" value={Math.round(average).toLocaleString()} />
          <Stat label="Top-500k line (avg)" value={Math.round(targetAverage).toLocaleString()} sub={`${signed(average - targetAverage, 0)} a season`} />
          <Stat label="Seasons over the line" value={`${seasons.filter((season) => season.targetHit).length} of ${seasons.length}`} />
        </div>
        <div className="backtest-chart" role="img" aria-label="Season points against the top-500k line">
          {seasons.map((season) => {
            const height = (value: number) => `${((value - minPoints) / (maxPoints - minPoints)) * 100}%`;
            return (
              <div className="backtest-col" key={season.season} title={`${season.season}: ${season.points} pts; top-500k ≈ ${season.top500Target}`}>
                <div className="backtest-bar-area">
                  <i className={season.targetHit ? "hit" : ""} style={{ height: height(season.points) }} />
                  <b style={{ bottom: height(season.top500Target) }} />
                </div>
                <span className="backtest-value">{season.points}</span>
                <span className={`backtest-margin ${season.targetMargin >= 0 ? "pos-val" : "neg-val"}`}>{signed(season.targetMargin, 0)}</span>
                <span className="backtest-label">{season.season.slice(2)}</span>
              </div>
            );
          })}
        </div>
        <div className="table-scroll">
          <table className="data-table compact">
            <thead>
              <tr><th>Season</th><th className="num">Points</th><th className="num">Top-500k line</th><th className="num">Margin</th><th className="num">Transfers</th><th>Chips</th></tr>
            </thead>
            <tbody>
              {seasons.map((season) => (
                <tr key={season.season}>
                  <td>{season.season}</td>
                  <td className="num strong">{season.points}</td>
                  <td className="num">{season.top500Target}</td>
                  <td className={`num ${season.targetMargin >= 0 ? "pos-val" : "neg-val"}`}>{signed(season.targetMargin, 0)}</td>
                  <td className="num">{season.transfers}</td>
                  <td className="small">{season.chips.map((chip) => `${chip.chip} GW${chip.gw}`).join(", ") || "—"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>

      <Card title="How the projections work">
        <ul className="plain-list">
          <li><strong>Minutes first.</strong> Each player gets a chance of starting, coming off the bench or missing out, from recent line-ups, rotation and official injury news.</li>
          <li><strong>Points by route.</strong> Goals, assists, clean sheets, defensive contributions, bonus and appearance points are forecast separately and added up.</li>
          <li><strong>Team strength and fixtures.</strong> Each club&apos;s attack and defence are rated from results and expected goals, blended with betting markets, and applied to the actual opponent.</li>
          <li><strong>Decisions.</strong> Transfers, captain and chips are chosen by an optimiser under the real FPL rules, with thresholds tested on past seasons.</li>
          <li><strong>Limits.</strong> Late team news after the data refresh is not included, so always check the news before the deadline.</li>
        </ul>
        <p className="note">
          Data: {DATA_SOURCES.map((source, index) => (
            <span key={source.url}>{index > 0 && ", "}<a href={source.url} target="_blank" rel="noreferrer">{source.label}</a></span>
          ))}. Model output generated {new Date(results.generatedAt).toLocaleString("en-GB")}.
        </p>
      </Card>
    </div>
  );
}

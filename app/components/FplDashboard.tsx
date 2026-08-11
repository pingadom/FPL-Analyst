"use client";

import { useEffect, useMemo, useState } from "react";
import results from "../data/model-results.json";

type WeightKey =
  | "performance"
  | "value"
  | "age"
  | "fixture"
  | "crowd"
  | "minutes"
  | "underlying";

type Player = {
  id: number;
  name: string;
  team: string;
  position: "GK" | "DEF" | "MID" | "FWD";
  price: number;
  ownership: number;
  projected: number;
  captainRating: number;
  score: number;
  opponent: string;
  venue: string;
  starter: boolean;
  captain: boolean;
  vice: boolean;
  trend: "up" | "down" | "flat";
  features: {
    recent: number;
    history: number;
    recentValue: number;
    historyValue: number;
    age: number;
    fixture: number;
    crowd: number;
    minutes: number;
    underlying: number;
  };
};

type ScoredPlayer = Player & { liveScore: number };

const positionOrder = ["GK", "DEF", "MID", "FWD"] as const;
const positionQuota: Record<Player["position"], number> = {
  GK: 2,
  DEF: 5,
  MID: 5,
  FWD: 3,
};

const weightLabels: Record<WeightKey, { label: string; hint: string }> = {
  performance: { label: "Performance", hint: "Points signal" },
  value: { label: "Value", hint: "Output per £m" },
  age: { label: "Age curve", hint: "Reliability prior" },
  fixture: { label: "Fixture", hint: "Next 4 opponents" },
  crowd: { label: "Market", hint: "Ownership signal" },
  minutes: { label: "Minutes", hint: "60-minute security" },
  underlying: { label: "Underlying", hint: "ICT involvement" },
};

function rebalanceWeights(
  current: Record<WeightKey, number>,
  changed: WeightKey,
  value: number,
) {
  const next = { ...current, [changed]: value };
  const otherKeys = (Object.keys(current) as WeightKey[]).filter(
    (key) => key !== changed,
  );
  const otherTotal = otherKeys.reduce((sum, key) => sum + current[key], 0);
  const remaining = 100 - value;
  if (otherTotal === 0) {
    otherKeys.forEach((key) => (next[key] = remaining / otherKeys.length));
  } else {
    otherKeys.forEach(
      (key) => (next[key] = (current[key] / otherTotal) * remaining),
    );
  }
  const rounded = Object.fromEntries(
    Object.entries(next).map(([key, amount]) => [key, Math.round(amount)]),
  ) as Record<WeightKey, number>;
  const difference = 100 - Object.values(rounded).reduce((a, b) => a + b, 0);
  const correction = otherKeys.find((key) => rounded[key] + difference >= 0);
  if (correction) rounded[correction] += difference;
  return rounded;
}

function calculateScore(
  player: Player,
  weights: Record<WeightKey, number>,
  recentShare: number,
) {
  const recent = recentShare / 100;
  const performance =
    player.features.recent * recent + player.features.history * (1 - recent);
  const value =
    player.features.recentValue * recent +
    player.features.historyValue * (1 - recent);
  return (
    performance * (weights.performance / 100) +
    value * (weights.value / 100) +
    player.features.age * (weights.age / 100) +
    player.features.fixture * (weights.fixture / 100) +
    player.features.crowd * (weights.crowd / 100) +
    player.features.minutes * (weights.minutes / 100) +
    player.features.underlying * (weights.underlying / 100)
  );
}

function buildSquad(players: ScoredPlayer[]) {
  let best: ScoredPlayer[] = [];
  let bestScore = -Infinity;

  for (let step = 0; step <= 120; step += 1) {
    const pricePenalty = step * 0.0002;
    const sorted = [...players].sort(
      (a, b) =>
        b.liveScore - b.price * pricePenalty -
        (a.liveScore - a.price * pricePenalty),
    );
    const selected: ScoredPlayer[] = [];
    const positionCount: Partial<Record<Player["position"], number>> = {};
    const teamCount: Record<string, number> = {};

    for (const player of sorted) {
      if ((positionCount[player.position] ?? 0) >= positionQuota[player.position])
        continue;
      if ((teamCount[player.team] ?? 0) >= 3) continue;
      selected.push(player);
      positionCount[player.position] = (positionCount[player.position] ?? 0) + 1;
      teamCount[player.team] = (teamCount[player.team] ?? 0) + 1;
      if (selected.length === 15) break;
    }
    if (selected.length !== 15) continue;
    const spend = selected.reduce((sum, player) => sum + player.price, 0);
    if (spend > 100) continue;
    const score = selected.reduce((sum, player) => sum + player.liveScore, 0);
    if (score > bestScore) {
      bestScore = score;
      best = selected;
    }
  }

  if (best.length === 0) {
    const originalIds = new Set((results.squad as Player[]).map((player) => player.id));
    best = players.filter((player) => originalIds.has(player.id));
  }

  let bestXi: ScoredPlayer[] = [];
  let bestXiScore = -Infinity;
  for (const defenders of [3, 4, 5]) {
    for (const forwards of [1, 2, 3]) {
      const midfielders = 10 - defenders - forwards;
      if (midfielders < 2 || midfielders > 5) continue;
      const formation: Record<Player["position"], number> = {
        GK: 1,
        DEF: defenders,
        MID: midfielders,
        FWD: forwards,
      };
      const xi = positionOrder.flatMap((position) =>
        best
          .filter((player) => player.position === position)
          .sort((a, b) => b.liveScore - a.liveScore)
          .slice(0, formation[position]),
      );
      const score = xi.reduce((sum, player) => sum + player.liveScore, 0);
      if (xi.length === 11 && score > bestXiScore) {
        bestXiScore = score;
        bestXi = xi;
      }
    }
  }

  return { squad: best, xi: bestXi };
}

function formatCountdown(deadline: string, now: number) {
  const difference = Math.max(0, new Date(deadline).getTime() - now);
  const days = Math.floor(difference / 86_400_000);
  const hours = Math.floor((difference % 86_400_000) / 3_600_000);
  const minutes = Math.floor((difference % 3_600_000) / 60_000);
  return `${days}d ${hours}h ${minutes}m`;
}

function PlayerRow({
  player,
  captain,
  vice,
}: {
  player: ScoredPlayer;
  captain?: boolean;
  vice?: boolean;
}) {
  return (
    <div className="player-row">
      <div className={`position-tag position-${player.position.toLowerCase()}`}>
        {player.position}
      </div>
      <div className="player-main">
        <div className="player-name-line">
          <span className="player-name">{player.name}</span>
          {captain && <span className="captain-chip">C</span>}
          {vice && <span className="vice-chip">V</span>}
        </div>
        <span className="player-meta">
          {player.team} · {player.opponent} ({player.venue}) · {player.ownership}%
        </span>
      </div>
      <div className="player-price">£{player.price.toFixed(1)}</div>
      <div className="player-projection">
        <strong>{player.projected.toFixed(1)}</strong>
        <span>xPts</span>
      </div>
    </div>
  );
}

export default function FplDashboard() {
  const calibrated = results.model.weights;
  const [weights, setWeights] = useState<Record<WeightKey, number>>({
    performance: calibrated.performance,
    value: calibrated.value,
    age: calibrated.age,
    fixture: calibrated.fixture,
    crowd: calibrated.crowd,
    minutes: calibrated.minutes,
    underlying: calibrated.underlying,
  });
  const [recentShare, setRecentShare] = useState(calibrated.recent);
  const [now, setNow] = useState(() => Date.now());
  const [showBench, setShowBench] = useState(true);

  useEffect(() => {
    const timer = window.setInterval(() => setNow(Date.now()), 60_000);
    return () => window.clearInterval(timer);
  }, []);

  const scoredPlayers = useMemo(
    () =>
      (results.currentPlayers as Player[]).map((player) => ({
        ...player,
        liveScore: calculateScore(player, weights, recentShare),
      })),
    [weights, recentShare],
  );
  const selection = useMemo(() => buildSquad(scoredPlayers), [scoredPlayers]);
  const xiIds = useMemo(
    () => new Set(selection.xi.map((player) => player.id)),
    [selection.xi],
  );
  const starters = useMemo(
    () =>
      [...selection.xi].sort(
        (a, b) => positionOrder.indexOf(a.position) - positionOrder.indexOf(b.position),
      ),
    [selection.xi],
  );
  const bench = useMemo(
    () =>
      selection.squad
        .filter((player) => !xiIds.has(player.id))
        .sort((a, b) => b.liveScore - a.liveScore),
    [selection.squad, xiIds],
  );
  const captainOrder = useMemo(
    () => [...starters].sort((a, b) => b.captainRating - a.captainRating),
    [starters],
  );
  const captain = captainOrder[0];
  const vice = captainOrder[1];
  const spend = selection.squad.reduce((sum, player) => sum + player.price, 0);
  const projected =
    starters.reduce((sum, player) => sum + player.projected, 0) +
    (captain?.projected ?? 0);
  const formation = positionOrder
    .slice(1)
    .map((position) => starters.filter((player) => player.position === position).length)
    .join("-");
  const topRanks = useMemo(
    () => [...scoredPlayers].sort((a, b) => b.liveScore - a.liveScore).slice(0, 6),
    [scoredPlayers],
  );
  const averageUplift =
    results.backtest.reduce((sum, season) => sum + season.uplift, 0) /
    results.backtest.length;

  const resetModel = () => {
    setWeights({
      performance: calibrated.performance,
      value: calibrated.value,
      age: calibrated.age,
      fixture: calibrated.fixture,
      crowd: calibrated.crowd,
      minutes: calibrated.minutes,
      underlying: calibrated.underlying,
    });
    setRecentShare(calibrated.recent);
  };

  return (
    <main>
      <header className="site-header">
        <a className="brand" href="#top" aria-label="FPL Lens home">
          <span className="brand-mark">FL</span>
          <span>FPL LENS</span>
          <span className="live-pill"><i /> LIVE</span>
        </a>
        <nav aria-label="Primary navigation">
          <a href="#squad">Squad</a>
          <a href="#backtest">Backtest</a>
          <a href="#method">Method</a>
        </nav>
        <div className="deadline-chip">
          <span>GW{results.headline.gameweek} LOCK</span>
          <strong>{formatCountdown(results.headline.deadline, now)}</strong>
        </div>
      </header>

      <section className="hero" id="top">
        <div className="eyebrow"><span>01</span> DECISION ROOM · {results.headline.season}</div>
        <div className="hero-copy">
          <h1>Build a squad<br />you can defend.</h1>
          <p>
            {results.model.trials.toLocaleString()} candidate mixes. {results.model.recursiveTrials} finalists.
            One legal squad carried forward and changed at every historical deadline.
          </p>
        </div>
        <div className="hero-metrics">
          <div><span>Projected</span><strong>{projected.toFixed(1)}</strong><small>GW points</small></div>
          <div><span>Budget</span><strong>£{spend.toFixed(1)}</strong><small>of £100m</small></div>
          <div><span>Shape</span><strong>{formation}</strong><small>best XI</small></div>
          <div><span>Backtest</span><strong>+{averageUplift.toFixed(1)}%</strong><small>vs baseline</small></div>
        </div>
      </section>

      <div className="ticker" aria-label="Model status">
        <span>MODEL {results.model.version}</span>
        <span>{results.model.recursiveTrials} RECURSIVE FINALISTS</span>
        <span>{results.model.playerWeeks.toLocaleString()} PLAYER-WEEKS</span>
        <span>{results.currentMeta.playersScored} CURRENT PLAYERS SCORED</span>
        <span>LAST REFRESH {new Date(results.generatedAt).toLocaleDateString("en-GB")}</span>
      </div>

      <section className="decision-grid" id="squad">
        <aside className="control-panel">
          <div className="section-label"><span>01</span> MODEL MIX</div>
          <div className="panel-heading">
            <div><h2>Tune the lens</h2><p>Weights always total 100%.</p></div>
            <button className="text-button" onClick={resetModel}>Reset</button>
          </div>
          <div className="sliders">
            {(Object.keys(weightLabels) as WeightKey[]).map((key) => (
              <label className="slider-control" key={key}>
                <span><strong>{weightLabels[key].label}</strong><small>{weightLabels[key].hint}</small></span>
                <output>{weights[key]}%</output>
                <input
                  type="range"
                  min="0"
                  max="100"
                  value={weights[key]}
                  aria-label={`${weightLabels[key].label} weight`}
                  onChange={(event) =>
                    setWeights((current) =>
                      rebalanceWeights(current, key, Number(event.target.value)),
                    )
                  }
                />
              </label>
            ))}
          </div>
          <div className="memory-control">
            <div className="memory-title"><strong>Form memory</strong><span>{recentShare}% recent</span></div>
            <input
              type="range"
              min="10"
              max="95"
              value={recentShare}
              aria-label="Recent performance share"
              onChange={(event) => setRecentShare(Number(event.target.value))}
            />
            <div className="memory-scale"><span>Season history</span><span>Last 4 GWs</span></div>
          </div>
          <div className="model-note">
            <span className="pulse-dot" />
            <p><strong>Calibrated preset</strong> won trial #{results.model.bestTrial} in the recursive replay. Move any control to stress-test it.</p>
          </div>
        </aside>

        <section className="squad-panel">
          <div className="section-label"><span>02</span> OPTIMAL XV</div>
          <div className="panel-heading squad-heading">
            <div><h2>Gameweek {results.headline.gameweek} squad</h2><p>Legal budget · max 3 per club · live re-optimisation</p></div>
            <div className="captain-call"><span>Captain</span><strong>{captain?.name ?? "—"}</strong></div>
          </div>
          <div className="squad-column-labels"><span>Player / fixture</span><span>Price</span><span>Forecast</span></div>
          <div className="squad-list">
            {starters.map((player) => (
              <PlayerRow
                key={player.id}
                player={player}
                captain={player.id === captain?.id}
                vice={player.id === vice?.id}
              />
            ))}
          </div>
          <button className="bench-toggle" onClick={() => setShowBench((value) => !value)} aria-expanded={showBench}>
            <span>BENCH · {bench.map((player) => player.name).join(" / ")}</span>
            <span>{showBench ? "−" : "+"}</span>
          </button>
          {showBench && (
            <div className="bench-list">
              {bench.map((player) => <PlayerRow key={player.id} player={player} />)}
            </div>
          )}
        </section>

        <aside className="edge-panel">
          <div className="section-label"><span>03</span> MARKET EDGES</div>
          <div className="panel-heading"><div><h2>Model board</h2><p>Who rises under your mix.</p></div></div>
          <div className="rank-list">
            {topRanks.map((player, index) => (
              <div className="rank-row" key={player.id}>
                <span className="rank-number">{String(index + 1).padStart(2, "0")}</span>
                <div><strong>{player.name}</strong><small>{player.team} · {player.opponent} ({player.venue})</small></div>
                <div className="rank-score"><strong>{Math.round(player.liveScore * 100)}</strong><span>lens</span></div>
              </div>
            ))}
          </div>
          <div className="fixture-edge-card">
            <div className="mini-label">FIXTURE DISAGREEMENT</div>
            {results.fixtureMatchups.slice(0, 3).map((match) => (
              <div className="fixture-edge" key={match.fixture}>
                <span>{match.fixture}</span>
                <strong>{match.modelPick}</strong>
                <small>{match.modelPick === match.popularPick ? "Model agrees with market" : `Market: ${match.popularPick}`}</small>
              </div>
            ))}
          </div>
        </aside>
      </section>

      <section className="backtest-section" id="backtest">
        <div className="backtest-intro">
          <div className="section-label light"><span>04</span> PROOF, NOT PROMISES</div>
          <h2>Replay the past.<br />Earn the present.</h2>
          <p>
            The same 15-player squad moves from one deadline to the next. The model
            makes a data-led transfer, selects a legal XI, orders the bench and then
            scores autosubs plus captaincy. No future result enters the decision.
          </p>
          <div className="proof-stat"><strong>{results.model.trials.toLocaleString()}</strong><span>candidate weight mixes</span></div>
          <div className="proof-stat"><strong>{results.model.recursiveTrials}</strong><span>full recursive finalists</span></div>
          <div className="proof-stat"><strong>{results.simulationSummary.averageWeeksChanged}</strong><span>average GWs changed / season</span></div>
        </div>
        <div className="season-chart" role="img" aria-label="Model and baseline points by season">
          {results.backtest.map((season) => {
            const modelHeight = Math.max(40, (season.points / 2200) * 100);
            const baselineHeight = Math.max(40, (season.baseline / 2200) * 100);
            return (
              <div className="season-column" key={season.season}>
                <div className="bar-area">
                  <div className="baseline-mark" style={{ height: `${baselineHeight}%` }} />
                  <div className="model-bar" style={{ height: `${modelHeight}%` }}>
                    <span>{season.points}</span>
                  </div>
                </div>
                <strong>{season.season}</strong>
                <span className={season.uplift >= 0 ? "positive" : "negative"}>
                  {season.uplift >= 0 ? "+" : ""}{season.uplift}%
                </span>
              </div>
            );
          })}
          <div className="chart-legend"><span><i className="legend-model" /> Recursive Lens 2.0</span><span><i className="legend-base" /> Recursive Lens 1.0</span></div>
        </div>
        <div className="expert-tests" aria-label="Tests of FPL champion advice">
          <div className="expert-tests-heading">
            <span>CHAMPION ADVICE, TESTED</span>
            <p>Average points per season versus the paired alternative. We keep what survives the replay.</p>
          </div>
          {results.expertTests.map((test) => (
            <article key={test.label} className={`expert-test ${test.result}`}>
              <div><span>{test.result}</span><strong>{test.delta > 0 ? "+" : ""}{test.delta}</strong><small>pts / season</small></div>
              <h3>{test.label}</h3>
              <p>{test.detail}</p>
            </article>
          ))}
        </div>
      </section>

      <section className="method-section" id="method">
        <div className="section-label"><span>05</span> HOW THE LENS WORKS</div>
        <div className="method-headline">
          <h2>Transparent inputs.<br />No mystery score.</h2>
          <p>{results.model.method} {results.model.objective}</p>
        </div>
        <div className="method-steps">
          <article><span>01</span><h3>Observe</h3><p>Points, price, minutes security, ICT involvement, market movement and the next four opponents.</p></article>
          <article><span>02</span><h3>Shift</h3><p>All rolling statistics move back one gameweek. The model never sees the result it is trying to predict.</p></article>
          <article><span>03</span><h3>Recurse</h3><p>{results.model.recursiveTrials} finalists carry a legal squad, bank and prices through every season deadline.</p></article>
          <article><span>04</span><h3>Optimise</h3><p>The best positive transfer changes the squad each week; your sliders rebuild today’s legal £100m squad live.</p></article>
        </div>
        <div className="method-footer">
          <div><span>AGE COVERAGE</span><strong>{Math.min(...results.dataSummary.map((item) => item.ageCoverage))}%+</strong></div>
          <div><span>TRAINING ROWS</span><strong>{results.model.playerWeeks.toLocaleString()}</strong></div>
          <div><span>CURRENT POOL</span><strong>{results.currentMeta.playersScored}</strong></div>
          <div className="source-links">
            {results.sources.map((source) => <a href={source.url} target="_blank" rel="noreferrer" key={source.url}>{source.label} ↗</a>)}
          </div>
        </div>
      </section>

      <footer>
        <div className="brand footer-brand"><span className="brand-mark">FL</span><span>FPL LENS</span></div>
        <p>Decision support, not certainty. Check late team news before every deadline.</p>
        <span>MODEL {results.model.version} · GENERATED {new Date(results.generatedAt).toLocaleString("en-GB")}</span>
      </footer>
    </main>
  );
}

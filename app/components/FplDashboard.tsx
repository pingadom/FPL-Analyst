"use client";

import { useEffect, useMemo, useState } from "react";
import results from "../data/model-results.json";

type WeightKey =
  | "performance"
  | "value"
  | "age"
  | "fixture"
  | "team"
  | "crowd"
  | "minutes"
  | "underlying";

type RiskMode = "protect" | "balanced" | "chase";

type Player = {
  id: number;
  name: string;
  team: string;
  position: "GK" | "DEF" | "MID" | "FWD";
  price: number;
  ownership: number;
  projected: number;
  sixWeekProjected: number;
  expectedMinutes: number;
  uncertainty: number;
  confidence: number;
  valueProjected: number;
  verdict: "Priority" | "Strong" | "Watch" | "Fade";
  setPieces: string[];
  riskFlags: string[];
  archetype: string;
  minutesModel: {
    startProbability: number;
    playProbability: number;
    sixtyProbability: number;
    minutesIfStart: number;
    minutesIfBench: number;
    minutesStd: number;
    rotationVolatility: number;
  };
  distribution: {
    p10: number;
    median: number;
    p90: number;
    blankProbability: number;
    return5Probability: number;
    haul8Probability: number;
    standardDeviation: number;
  };
  defenderModel: {
    actionRate: number;
    contributionProbability: number;
    bpsRate: number;
    goalRoute: number;
    assistRoute: number;
  };
  ensemble: {
    structural: number;
    empirical: number;
    marketRole: number;
    official: number;
    disagreement: number;
  };
  marketForecast: {
    priceRiseProbability: number;
    priceFallProbability: number;
  };
  components: {
    appearance: number;
    goals: number;
    assists: number;
    cleanSheet: number;
    defence: number;
    bonus: number;
    adjustment: number;
  };
  history: {
    matches: number;
    average: number;
    per90: number;
    returnRate: number;
    volatility: number;
  };
  opponentHistory: {
    matches: number;
    average: number;
    per90: number;
    returnRate: number;
  };
  teamContext: {
    expectedGoalsFor: number;
    expectedGoalsAgainst: number;
    cleanSheetProbability: number;
    horizonExpectedGoalsAgainst: number;
    horizonCleanSheetProbability: number;
    attackRank: number;
    defenceRank: number;
    strengthRank: number;
    ratingConfidence: number;
    regimeShift: number;
  };
  comparison: {
    fixtureRank: number;
    fixturePlayers: number;
    positionRank: number;
    positionPlayers: number;
    projectionRank: number;
    popularRival: string;
    popularRivalOwnership: number;
    popularRivalProjection: number;
    edgeVsPopular: number;
  };
  captainRating: number;
  score: number;
  strategyScores: Record<RiskMode, number>;
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
    team: number;
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
  fixture: { label: "Fixture", hint: "Next 6 opponents" },
  team: { label: "Team strength", hint: "Attack + defence" },
  crowd: { label: "Market", hint: "Ownership signal" },
  minutes: { label: "Minutes", hint: "60-minute security" },
  underlying: { label: "Underlying", hint: "ICT involvement" },
};

const chipDescriptions: Record<string, string> = {
  Wildcard: "Persistent squad gap, with extra urgency when AFCON removes multiple players.",
  "Free Hit": "Reserved for major blank clashes or unusually concentrated double fixtures.",
  "Bench Boost": "Eligible when the bench contains a double-gameweek player and clears the score bar.",
  "Triple Captain": "Eligible only when the chosen captain has two fixtures and clears the score bar.",
};

const componentLabels: Record<keyof Player["components"], string> = {
  appearance: "Appearance",
  goals: "Goal threat",
  assists: "Assist threat",
  cleanSheet: "Clean sheet",
  defence: "Defensive work",
  bonus: "Bonus",
  adjustment: "Model adjustment",
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
  riskMode: RiskMode,
) {
  const recent = recentShare / 100;
  const performance =
    player.features.recent * recent + player.features.history * (1 - recent);
  const value =
    player.features.recentValue * recent +
    player.features.historyValue * (1 - recent);
  const lensScore = (
    performance * (weights.performance / 100) +
    value * (weights.value / 100) +
    player.features.age * (weights.age / 100) +
    player.features.fixture * (weights.fixture / 100) +
    player.features.team * (weights.team / 100) +
    player.features.crowd * (weights.crowd / 100) +
    player.features.minutes * (weights.minutes / 100) +
    player.features.underlying * (weights.underlying / 100)
  );
  return (
    0.58 * lensScore +
    0.14 * (player.comparison.projectionRank / 100) +
    0.10 * (player.confidence / 100) +
    0.18 * player.strategyScores[riskMode]
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
    const score = selected.reduce(
      (sum, player) =>
        sum + 0.72 * player.liveScore + 0.20 * player.sixWeekProjected / 30 +
        0.08 * player.confidence / 100,
      0,
    );
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
  onAnalyse,
}: {
  player: ScoredPlayer;
  captain?: boolean;
  vice?: boolean;
  onAnalyse?: (player: ScoredPlayer) => void;
}) {
  return (
    <button
      className="player-row"
      onClick={() => onAnalyse?.(player)}
      type="button"
      aria-label={`Analyse ${player.name}`}
    >
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
          {player.team} · {player.opponent} ({player.venue}) · {player.minutesModel.startProbability}% start · {player.distribution.return5Probability}% return
        </span>
      </div>
      <div className="player-price">£{player.price.toFixed(1)}</div>
      <div className="player-projection">
        <strong>{player.projected.toFixed(1)}</strong>
        <span>xPts</span>
      </div>
    </button>
  );
}

export default function FplDashboard() {
  const calibrated = results.model.weights as typeof results.model.weights & { team: number };
  const [weights, setWeights] = useState<Record<WeightKey, number>>({
    performance: calibrated.performance,
    value: calibrated.value,
    age: calibrated.age,
    fixture: calibrated.fixture,
    team: calibrated.team ?? 0,
    crowd: calibrated.crowd,
    minutes: calibrated.minutes,
    underlying: calibrated.underlying,
  });
  const [recentShare, setRecentShare] = useState(calibrated.recent);
  const [riskMode, setRiskMode] = useState<RiskMode>("balanced");
  const [now, setNow] = useState(() => Date.now());
  const [showBench, setShowBench] = useState(true);
  const [analysedId, setAnalysedId] = useState<number>(
    (results.currentPlayers as Player[])[0]?.id ?? 0,
  );

  useEffect(() => {
    const timer = window.setInterval(() => setNow(Date.now()), 60_000);
    return () => window.clearInterval(timer);
  }, []);

  const scoredPlayers = useMemo(
    () =>
      (results.currentPlayers as Player[]).map((player) => ({
        ...player,
        liveScore: calculateScore(player, weights, recentShare, riskMode),
      })),
    [weights, recentShare, riskMode],
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
  const analysedPlayer = useMemo(
    () =>
      scoredPlayers.find((player) => player.id === analysedId) ??
      scoredPlayers[0],
    [analysedId, scoredPlayers],
  );
  const resetModel = () => {
    setWeights({
      performance: calibrated.performance,
      value: calibrated.value,
      age: calibrated.age,
      fixture: calibrated.fixture,
      team: calibrated.team ?? 0,
      crowd: calibrated.crowd,
      minutes: calibrated.minutes,
      underlying: calibrated.underlying,
    });
    setRecentShare(calibrated.recent);
    setRiskMode("balanced");
  };
  const openAnalysis = (player: ScoredPlayer) => {
    setAnalysedId(player.id);
    window.requestAnimationFrame(() =>
      document.getElementById("player-lab")?.scrollIntoView({
        behavior: "smooth",
        block: "start",
      }),
    );
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
          <a href="#player-lab">Player Lab</a>
          <a href="#chips">Chips</a>
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
          <div><span>P(70+)</span><strong>{results.headline.scenario.probability70}%</strong><small>{results.headline.scenario.p10}–{results.headline.scenario.p90} scenario range</small></div>
        </div>
      </section>

      <div className="ticker" aria-label="Model status">
        <span>MODEL {results.model.version}</span>
        <span>{results.model.recursiveTrials} RECURSIVE FINALISTS</span>
        <span>{results.chipStrategy.policyTrials} CHIP POLICIES</span>
        <span>{results.rankTarget.averageProbability}% AVG TARGET PROBABILITY</span>
        <span>{results.model.playerWeeks.toLocaleString()} PLAYER-WEEKS</span>
        <span>{results.currentMeta.playersScored} CURRENT PLAYERS SCORED</span>
        <span>{results.headline.scenario.simulations.toLocaleString()} CORRELATED SQUAD SCENARIOS</span>
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
          <div className="risk-control">
            <div className="memory-title"><strong>Decision profile</strong><span>{riskMode}</span></div>
            <div className="risk-buttons" role="group" aria-label="Squad risk profile">
              {(["protect", "balanced", "chase"] as RiskMode[]).map((mode) => (
                <button
                  type="button"
                  className={riskMode === mode ? "active" : ""}
                  aria-pressed={riskMode === mode}
                  onClick={() => setRiskMode(mode)}
                  key={mode}
                >
                  {mode}
                </button>
              ))}
            </div>
            <p>Protect weights downside and minutes; Chase weights 8+ point probability and upside.</p>
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
                onAnalyse={openAnalysis}
              />
            ))}
          </div>
          <button className="bench-toggle" onClick={() => setShowBench((value) => !value)} aria-expanded={showBench}>
            <span>BENCH · {bench.map((player) => player.name).join(" / ")}</span>
            <span>{showBench ? "−" : "+"}</span>
          </button>
          {showBench && (
            <div className="bench-list">
              {bench.map((player) => <PlayerRow key={player.id} player={player} onAnalyse={openAnalysis} />)}
            </div>
          )}
        </section>

        <aside className="edge-panel">
          <div className="section-label"><span>03</span> MARKET EDGES</div>
          <div className="panel-heading"><div><h2>Model board</h2><p>Who rises under your mix.</p></div></div>
          <div className="rank-list">
            {topRanks.map((player, index) => (
              <button className="rank-row" key={player.id} type="button" onClick={() => openAnalysis(player)}>
                <span className="rank-number">{String(index + 1).padStart(2, "0")}</span>
                <div><strong>{player.name}</strong><small>{player.team} · {player.opponent} ({player.venue})</small></div>
                <div className="rank-score"><strong>{Math.round(player.liveScore * 100)}</strong><span>lens</span></div>
              </button>
            ))}
          </div>
          <div className="fixture-edge-card">
            <div className="mini-label">FIXTURE DISAGREEMENT</div>
            {results.fixtureMatchups.slice(0, 3).map((match) => (
              <div className="fixture-edge" key={match.fixture}>
                <span>{match.fixture}</span>
                <strong>{match.modelPick}</strong>
                <small>
                  {match.modelPick === match.popularPick
                    ? `${match.modelProjection} xPts · model agrees with market`
                    : `${match.modelProjection} xPts vs ${match.popularPick} ${match.popularProjection} · ${match.popularOwnership}% owned`}
                </small>
              </div>
            ))}
          </div>
        </aside>
      </section>

      {analysedPlayer && (
        <section className="player-lab" id="player-lab">
          <div className="player-lab-intro">
            <div className="section-label"><span>04</span> PLAYER LAB</div>
            <h2>Read the player.<br />Not just the score.</h2>
            <p>
              Open the full distribution: start and 60-minute probability, scoring
              routes, defender contributions, ensemble disagreement, team context
              and the strongest popular alternative in the same fixture.
            </p>
            <label className="player-picker">
              <span>PLAYER TO ANALYSE</span>
              <select
                value={analysedPlayer.id}
                onChange={(event) => setAnalysedId(Number(event.target.value))}
              >
                {[...scoredPlayers]
                  .sort((a, b) => b.sixWeekProjected - a.sixWeekProjected)
                  .map((player) => (
                    <option value={player.id} key={player.id}>
                      {player.name} · {player.team} · £{player.price.toFixed(1)}
                    </option>
                  ))}
              </select>
            </label>
            <div className="lab-method-note">
              <strong>{results.currentMeta.componentModel}</strong>
              <span>{results.currentMeta.historicalSeasons} historical season{results.currentMeta.historicalSeasons === 1 ? "" : "s"} in this player refresh.</span>
            </div>
          </div>

          <div className="player-workbench">
            <header className="player-lab-header">
              <div>
                <span>{analysedPlayer.position} · {analysedPlayer.team} · £{analysedPlayer.price.toFixed(1)}</span>
                <h3>{analysedPlayer.name}</h3>
                <p>{analysedPlayer.opponent} ({analysedPlayer.venue}) · {analysedPlayer.ownership}% owned</p>
              </div>
              <div className={`verdict verdict-${analysedPlayer.verdict.toLowerCase()}`}>
                <span>MODEL CALL</span>
                <strong>{analysedPlayer.verdict}</strong>
                <small>{analysedPlayer.confidence}% confidence</small>
              </div>
            </header>

            <div className="lab-metrics">
              <div><span>GW xPTS</span><strong>{analysedPlayer.projected.toFixed(1)}</strong><small>risk-aware forecast</small></div>
              <div><span>SIX-GW xPTS</span><strong>{analysedPlayer.sixWeekProjected.toFixed(1)}</strong><small>weighted horizon</small></div>
              <div><span>EXPECTED MINS</span><strong>{analysedPlayer.expectedMinutes}</strong><small>{analysedPlayer.minutesModel.startProbability}% start · {analysedPlayer.minutesModel.sixtyProbability}% 60+</small></div>
              <div><span>VALUE</span><strong>{analysedPlayer.valueProjected.toFixed(2)}</strong><small>six-GW xPts / £m</small></div>
            </div>

            <div className="lab-analysis-grid">
              <article className="component-card">
                <div className="lab-card-heading"><span>PROJECTION ANATOMY</span><strong>{analysedPlayer.projected.toFixed(1)} total</strong></div>
                <div className="component-bars">
                  {(Object.keys(componentLabels) as Array<keyof Player["components"]>).map((key) => {
                    const value = analysedPlayer.components[key];
                    const width = Math.min(100, Math.abs(value) / Math.max(analysedPlayer.projected, 0.5) * 100);
                    return (
                      <div className={`component-row ${value < 0 ? "component-negative" : ""}`} key={key}>
                        <span>{componentLabels[key]}</span>
                        <div><i style={{ width: `${width}%` }} /></div>
                        <strong>{value >= 0 ? "+" : ""}{value.toFixed(2)}</strong>
                      </div>
                    );
                  })}
                </div>
              </article>

              <article className="history-card">
                <div className="lab-card-heading"><span>PERFORMANCE EVIDENCE</span><strong>{analysedPlayer.history.matches} apps</strong></div>
                <div className="history-split">
                  <div>
                    <span>GENERAL</span>
                    <strong>{analysedPlayer.history.average.toFixed(2)}</strong>
                    <small>points / app · {analysedPlayer.history.per90.toFixed(2)} per 90</small>
                    <em>{analysedPlayer.history.returnRate}% returned 5+ points</em>
                  </div>
                  <div>
                    <span>VS {analysedPlayer.opponent}</span>
                    <strong>{analysedPlayer.opponentHistory.matches > 0 ? analysedPlayer.opponentHistory.average.toFixed(2) : "—"}</strong>
                    <small>{analysedPlayer.opponentHistory.matches} prior appearance{analysedPlayer.opponentHistory.matches === 1 ? "" : "s"}</small>
                    <em>{analysedPlayer.opponentHistory.matches > 0 ? `${analysedPlayer.opponentHistory.returnRate}% returned 5+ points` : "No direct sample"}</em>
                  </div>
                </div>
                <p>Opponent history is descriptive and sample-labelled; it does not override the forward projection.</p>
              </article>

              <article className="comparison-card">
                <div className="lab-card-heading"><span>SAME-FIXTURE TEST</span><strong>#{analysedPlayer.comparison.fixtureRank}/{analysedPlayer.comparison.fixturePlayers}</strong></div>
                <div className="comparison-player">
                  <div><span>MODEL</span><strong>{analysedPlayer.name}</strong><small>{analysedPlayer.projected.toFixed(1)} xPts</small></div>
                  <b>VS</b>
                  <div><span>POPULAR ALTERNATIVE</span><strong>{analysedPlayer.comparison.popularRival}</strong><small>{analysedPlayer.comparison.popularRivalProjection.toFixed(1)} xPts · {analysedPlayer.comparison.popularRivalOwnership}% owned</small></div>
                </div>
                <div className={`comparison-edge ${analysedPlayer.comparison.edgeVsPopular >= 0 ? "positive" : "negative"}`}>
                  {analysedPlayer.comparison.edgeVsPopular >= 0 ? "+" : ""}{analysedPlayer.comparison.edgeVsPopular.toFixed(1)} xPts versus the popular alternative
                </div>
                <small>Position rank #{analysedPlayer.comparison.positionRank}/{analysedPlayer.comparison.positionPlayers}</small>
              </article>

              <article className="signal-card">
                <div className="lab-card-heading"><span>ROLE & RISK</span><strong>{analysedPlayer.confidence}%</strong></div>
                <div className="signal-group">
                  <span>SET PIECES</span>
                  <div>{analysedPlayer.setPieces.length ? analysedPlayer.setPieces.map((signal) => <i key={signal}>{signal}</i>) : <i>None flagged</i>}</div>
                </div>
                <div className="signal-group">
                  <span>RISK CHECK</span>
                  <div>{analysedPlayer.riskFlags.map((flag) => <i className={flag === "No major flag" ? "safe-signal" : "risk-signal"} key={flag}>{flag}</i>)}</div>
                </div>
                <p>Projection uncertainty: ±{Math.max(0.4, analysedPlayer.uncertainty * 2.4).toFixed(1)} points around the central GW estimate.</p>
              </article>

              <article className="probability-card">
                <div className="lab-card-heading"><span>POINT DISTRIBUTION</span><strong>±{analysedPlayer.distribution.standardDeviation.toFixed(1)}</strong></div>
                <div className="distribution-track" aria-label={`10th to 90th percentile: ${analysedPlayer.distribution.p10} to ${analysedPlayer.distribution.p90} points`}>
                  <i /><b style={{ left: `${Math.min(94, Math.max(6, analysedPlayer.distribution.median / Math.max(analysedPlayer.distribution.p90, 1) * 100))}%` }} />
                </div>
                <div className="distribution-labels"><span>P10 <strong>{analysedPlayer.distribution.p10}</strong></span><span>MEDIAN <strong>{analysedPlayer.distribution.median}</strong></span><span>P90 <strong>{analysedPlayer.distribution.p90}</strong></span></div>
                <div className="probability-grid">
                  <div><strong>{analysedPlayer.distribution.blankProbability}%</strong><span>blank ≤2</span></div>
                  <div><strong>{analysedPlayer.distribution.return5Probability}%</strong><span>return 5+</span></div>
                  <div><strong>{analysedPlayer.distribution.haul8Probability}%</strong><span>haul 8+</span></div>
                </div>
              </article>

              <article className="minutes-card">
                <div className="lab-card-heading"><span>MINUTES TREE</span><strong>{analysedPlayer.minutesModel.startProbability}% START</strong></div>
                <div className="probability-grid minutes-grid">
                  <div><strong>{analysedPlayer.minutesModel.playProbability}%</strong><span>plays</span></div>
                  <div><strong>{analysedPlayer.minutesModel.sixtyProbability}%</strong><span>reaches 60</span></div>
                  <div><strong>{analysedPlayer.minutesModel.minutesIfStart}</strong><span>mins if start</span></div>
                </div>
                <p>Bench appearance: {analysedPlayer.minutesModel.minutesIfBench} conditional minutes · rotation volatility {analysedPlayer.minutesModel.rotationVolatility}%.</p>
              </article>

              <article className="defender-card">
                <div className="lab-card-heading"><span>ROLE ENGINE</span><strong>{analysedPlayer.archetype}</strong></div>
                <div className="probability-grid">
                  <div><strong>{analysedPlayer.defenderModel.contributionProbability}%</strong><span>DC return</span></div>
                  <div><strong>{analysedPlayer.defenderModel.actionRate}</strong><span>actions / 90</span></div>
                  <div><strong>{analysedPlayer.defenderModel.bpsRate}</strong><span>BPS / match</span></div>
                </div>
                <p>Goal route {analysedPlayer.defenderModel.goalRoute.toFixed(3)} and assist route {analysedPlayer.defenderModel.assistRoute.toFixed(3)} per 90, before opponent and set-piece adjustments.</p>
              </article>

              <article className="ensemble-card">
                <div className="lab-card-heading"><span>ENSEMBLE & PRICE</span><strong>{analysedPlayer.ensemble.disagreement.toFixed(1)} disagreement</strong></div>
                <div className="ensemble-bars">
                  {(["structural", "empirical", "marketRole", "official"] as const).map((key) => (
                    <div key={key}><span>{key.replace("marketRole", "market role")}</span><i><b style={{ width: `${analysedPlayer.ensemble[key]}%` }} /></i><strong>{analysedPlayer.ensemble[key]}%</strong></div>
                  ))}
                </div>
                <p>Price move forecast: {analysedPlayer.marketForecast.priceRiseProbability}% rise · {analysedPlayer.marketForecast.priceFallProbability}% fall.</p>
              </article>

              <article className="team-card">
                <div className="lab-card-heading"><span>TEAM CONTEXT</span><strong>STRENGTH #{analysedPlayer.teamContext.strengthRank}/20</strong></div>
                <div className="team-context-grid">
                  <div><span>ATTACK</span><strong>#{analysedPlayer.teamContext.attackRank}</strong><small>{analysedPlayer.teamContext.expectedGoalsFor.toFixed(2)} expected goals</small></div>
                  <div><span>DEFENCE</span><strong>#{analysedPlayer.teamContext.defenceRank}</strong><small>{analysedPlayer.teamContext.expectedGoalsAgainst.toFixed(2)} expected conceded</small></div>
                  <div><span>NEXT-GW CS</span><strong>{analysedPlayer.teamContext.cleanSheetProbability}%</strong><small>Poisson probability</small></div>
                  <div><span>SIX-GW CS</span><strong>{analysedPlayer.teamContext.horizonCleanSheetProbability}%</strong><small>{analysedPlayer.teamContext.horizonExpectedGoalsAgainst.toFixed(2)} expected conceded / match</small></div>
                </div>
                <p>
                  Defender and goalkeeper forecasts inherit the team clean-sheet environment,
                  then add expected minutes, attacking routes, defensive contributions and bonus.
                  Team rating confidence: {analysedPlayer.teamContext.ratingConfidence}% · regime-shift signal {analysedPlayer.teamContext.regimeShift}%.
                </p>
              </article>
            </div>
          </div>
        </section>
      )}

      <section className="chip-section" id="chips">
        <div className="chip-intro">
          <div className="section-label"><span>05</span> CHIP DESK</div>
          <h2>Wait for the<br />fixture to bend.</h2>
          <p>
            Chips are scored inside the recursive replay. Each decision compares
            today’s edge with the discounted option value of known future blanks,
            doubles and the chip-window expiry.
          </p>
          <div className="current-chip-call">
            <span>GW{results.chipStrategy.current.gameweek} CALL</span>
            <strong>{results.chipStrategy.current.chip}</strong>
            <p>{results.chipStrategy.current.reason}</p>
            <small>{results.chipStrategy.current.nextReview}</small>
          </div>
        </div>
        <div className="chip-workbench">
          <div className="chip-scoreboard">
            <div><span>FULL REPLAY</span><strong>+{results.chipStrategy.averageGain}</strong><small>pts / season</small></div>
            <div><span>WALK-FORWARD</span><strong>+{results.chipStrategy.walkForwardAverageGain}</strong><small>pts / season</small></div>
            <div><span>POLICIES TESTED</span><strong>{results.chipStrategy.policyTrials}</strong><small>threshold mixes</small></div>
          </div>
          <div className="chip-cards">
            {results.chipStrategy.breakdown.map((chip) => (
              <article key={chip.chip}>
                <div className="chip-card-top"><span>{chip.chip}</span><strong>+{chip.averageGain}</strong></div>
                <p>{chipDescriptions[chip.chip]}</p>
                <small>{chip.uses} qualified uses · average immediate gain</small>
              </article>
            ))}
          </div>
          <div className="chip-history">
            <div className="chip-history-heading">
              <span>WALK-FORWARD DECISIONS</span>
              <p>Only information available before that deadline. Green seasons beat the identical no-chip run.</p>
            </div>
            {results.backtest.map((season) => (
              <div className="chip-season-row" key={season.season}>
                <strong>{season.season}</strong>
                <div>
                  {season.chips.length > 0 ? season.chips.map((chip) => (
                    <span className="history-chip" key={`${chip.chip}-${chip.gw}`}>
                      {chip.chip.replace("Triple Captain", "TC").replace("Bench Boost", "BB").replace("Free Hit", "FH").replace("Wildcard", "WC")} · GW{chip.gw}
                    </span>
                  )) : <span className="history-chip muted-chip">No qualifying play</span>}
                </div>
                <span className={season.chipPoints >= 0 ? "positive" : "negative"}>
                  {season.chipPoints >= 0 ? "+" : ""}{season.chipPoints} pts
                </span>
              </div>
            ))}
          </div>
          <p className="chip-rules">{results.chipStrategy.rules}</p>
        </div>
      </section>

      <section className="backtest-section" id="backtest">
        <div className="backtest-intro">
          <div className="section-label light"><span>06</span> PROOF, NOT PROMISES</div>
          <h2>Replay the past.<br />Earn the present.</h2>
          <p>
            The same 15-player squad moves from one deadline to the next. The model
            makes a data-led transfer, selects a legal XI, orders the bench and then
            scores autosubs, captaincy and qualified chip decisions. No future result
            enters the decision.
          </p>
          <div className="proof-stat"><strong>{results.model.trials.toLocaleString()}</strong><span>candidate weight mixes</span></div>
          <div className="proof-stat"><strong>{results.model.recursiveTrials}</strong><span>full recursive finalists</span></div>
          <div className="proof-stat"><strong>{results.simulationSummary.averageWeeksChanged}</strong><span>average GWs changed / season</span></div>
          <div className={`rank-target-card ${results.rankTarget.hitRate >= 75 ? "on-target" : "off-target"}`}>
            <span>TOP-500K CONSISTENCY TEST</span>
            <strong>{results.rankTarget.hits}/{results.rankTarget.seasons}</strong>
            <p>{results.rankTarget.hitRate}% of seasons cleared the estimated pace line · {results.rankTarget.averageProbability}% average bootstrap probability.</p>
            <small>{results.rankTarget.method}</small>
          </div>
        </div>
        <div className="season-chart" role="img" aria-label="Model and baseline points by season">
          {results.backtest.map((season) => {
            const modelHeight = Math.max(40, (season.points / 2300) * 100);
            const baselineHeight = Math.max(40, (season.baseline / 2300) * 100);
            const targetHeight = Math.max(40, (season.top500Target / 2300) * 100);
            return (
              <div className="season-column" key={season.season}>
                <div className="bar-area">
                  <div className="baseline-mark" style={{ height: `${baselineHeight}%` }} />
                  <div className="target-mark" style={{ height: `${targetHeight}%` }} title={`Estimated top-500k pace: ${season.top500Target}`} />
                  <div className="model-bar" style={{ height: `${modelHeight}%` }}>
                    <span>{season.points}</span>
                  </div>
                </div>
                <strong>{season.season}</strong>
                <span className={season.targetHit ? "positive" : "negative"} title={season.estimatedBand}>
                  {season.targetMargin >= 0 ? "+" : ""}{season.targetMargin} to target
                </span>
              </div>
            );
          })}
          <div className="chart-legend"><span><i className="legend-model" /> {results.model.version} + chips</span><span><i className="legend-base" /> Same model, no chips</span><span><i className="legend-target" /> Est. top-500k pace</span></div>
        </div>
        <div className="diagnostics-panel">
          <div className="expert-tests-heading">
            <span>PROBABILITY CALIBRATION</span>
            <p>Forecast quality is checked by scoring route, not hidden behind season points.</p>
          </div>
          <div className="diagnostic-metrics">
            <div><strong>{results.calibrationDiagnostics.returnBrier}</strong><span>5+ return Brier</span></div>
            <div><strong>{results.calibrationDiagnostics.minutes60Brier}</strong><span>60-minute Brier</span></div>
            <div><strong>{results.calibrationDiagnostics.cleanSheetBrier}</strong><span>clean-sheet Brier</span></div>
            <div><strong>{results.calibrationDiagnostics.p10P90Coverage}%</strong><span>P10–P90 coverage</span></div>
          </div>
          <div className="calibration-curve" aria-label="Predicted and observed five-point return rates">
            {results.calibrationDiagnostics.returnCalibration.map((bin) => (
              <div key={`${bin.forecast}-${bin.observed}`} title={`${bin.players.toLocaleString()} player-weeks`}>
                <i style={{ height: `${bin.forecast}%` }} /><b style={{ height: `${bin.observed}%` }} />
                <span>{bin.forecast}%</span>
              </div>
            ))}
          </div>
        </div>
        <div className="current-rules-panel">
          <div>
            <span>CURRENT-RULES COUNTERFACTUAL</span>
            <strong>{results.currentRulesReplay.averagePoints}</strong>
            <small>average points · {results.currentRulesReplay.eventCoverage}% exact defensive-event coverage</small>
            <p>{results.currentRulesReplay.method}</p>
          </div>
          <div className="current-rule-seasons">
            {results.currentRulesReplay.seasons.map((season) => (
              <div key={season.season}><span>{season.season}</span><strong>{season.points}</strong><small>{season.deltaVsHistoricalRules >= 0 ? "+" : ""}{season.deltaVsHistoricalRules}</small></div>
            ))}
          </div>
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
        <div className="section-label"><span>07</span> HOW THE LENS WORKS</div>
        <div className="method-headline">
          <h2>Transparent inputs.<br />No mystery score.</h2>
          <p>{results.model.method} {results.model.objective}</p>
        </div>
        <div className="method-steps">
          <article><span>01</span><h3>Distribute</h3><p>Start, bench, 60-minute, return, defender-contribution and clean-sheet probabilities create a full player outcome range.</p></article>
          <article><span>02</span><h3>Shift</h3><p>All rolling statistics move back one gameweek. The model never sees the result it is trying to predict.</p></article>
          <article><span>03</span><h3>Ensemble</h3><p>Structural, empirical and market-role forecasts are position-weighted by prior causal error; the live official projection is an independent fourth vote.</p></article>
          <article><span>04</span><h3>Simulate</h3><p>{results.headline.scenario.simulations.toLocaleString()} correlated squad scenarios and {results.chipStrategy.policyTrials} option-value chip policies compete under legal FPL constraints.</p></article>
        </div>
        <div className="method-footer">
          <div><span>AGE COVERAGE</span><strong>{Math.min(...results.dataSummary.map((item) => item.ageCoverage))}%+</strong></div>
          <div><span>CALIBRATION ROWS</span><strong>{results.model.playerWeeks.toLocaleString()}</strong></div>
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

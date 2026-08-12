const POSITION_QUOTA = { GK: 2, DEF: 5, MID: 5, FWD: 3 };
const FORMATIONS = [
  ...[3, 4, 5].flatMap((defenders) =>
    [1, 2, 3]
      .map((forwards) => ({
        GK: 1,
        DEF: defenders,
        MID: 10 - defenders - forwards,
        FWD: forwards,
      }))
      .filter(({ MID }) => MID >= 2 && MID <= 5),
  ),
];

export const BENCH_PREMIUM_LIMIT = 2.0;

function immediateUtility(player) {
  return (
    0.68 * Number(player.projected) +
    0.18 * (Number(player.sixWeekProjected) / 6) +
    0.14 * (Number(player.liveScore) * 5)
  );
}

function captainUtility(player) {
  const median = Number(player.distribution?.median ?? player.projected);
  const startProbability = Number(player.minutesModel?.startProbability ?? 100) / 100;
  return (
    0.72 * Number(player.projected) +
    0.18 * median +
    0.10 * (Number(player.liveScore) * 5)
  ) * (0.85 + 0.15 * startProbability);
}

function absenceDistribution(players) {
  let probabilities = [1];
  for (const player of players) {
    const absence = Math.max(
      0,
      Math.min(1, 1 - Number(player.minutesModel?.playProbability ?? 100) / 100),
    );
    const next = Array(probabilities.length + 1).fill(0);
    probabilities.forEach((probability, missed) => {
      next[missed] += probability * (1 - absence);
      next[missed + 1] += probability * absence;
    });
    probabilities = next;
  }
  return probabilities;
}

function probabilityAtLeast(distribution, count) {
  return distribution.slice(count).reduce((sum, probability) => sum + probability, 0);
}

function clubCounts(players) {
  const counts = new Map();
  for (const player of players) counts.set(player.team, (counts.get(player.team) ?? 0) + 1);
  return counts;
}

function isLegalSquad(squad) {
  if (squad.length !== 15 || new Set(squad.map((player) => player.id)).size !== 15)
    return false;
  if (squad.reduce((sum, player) => sum + Number(player.price), 0) > 100.0001)
    return false;
  if ([...clubCounts(squad).values()].some((count) => count > 3)) return false;
  return Object.entries(POSITION_QUOTA).every(
    ([position, quota]) => squad.filter((player) => player.position === position).length === quota,
  );
}

function positionalFloors(players) {
  return Object.fromEntries(
    Object.keys(POSITION_QUOTA).map((position) => [
      position,
      Math.min(...players.filter((player) => player.position === position).map((player) => Number(player.price))),
    ]),
  );
}

function benchPremium(bench, floors) {
  return bench.reduce(
    (sum, player) => sum + Math.max(0, Number(player.price) - floors[player.position]),
    0,
  );
}

function chooseLineup(squad) {
  let best = null;
  for (const formation of FORMATIONS) {
    const xi = Object.entries(formation).flatMap(([position, count]) =>
      squad
        .filter((player) => player.position === position)
        .sort((a, b) => immediateUtility(b) - immediateUtility(a))
        .slice(0, count),
    );
    if (xi.length !== 11) continue;
    const captain = [...xi].sort((a, b) => captainUtility(b) - captainUtility(a))[0];
    const utility =
      xi.reduce((sum, player) => sum + immediateUtility(player), 0) + captainUtility(captain);
    if (!best || utility > best.utility) best = { xi, captain, utility };
  }
  return best;
}

export function evaluateSquad(squad, playerPool, options = {}) {
  if (!isLegalSquad(squad)) return null;
  const spend = squad.reduce((sum, player) => sum + Number(player.price), 0);
  const minimumSpend = Number(options.minimumSpend ?? 99.5);
  if (!options.allowStrategicBank && spend < minimumSpend - 1e-6) return null;
  const lineup = chooseLineup(squad);
  if (!lineup) return null;

  const xiIds = new Set(lineup.xi.map((player) => player.id));
  const bench = squad.filter((player) => !xiIds.has(player.id));
  const benchGoalkeeper = bench.find((player) => player.position === "GK");
  const outfieldBench = bench
    .filter((player) => player.position !== "GK")
    .sort((a, b) => immediateUtility(b) - immediateUtility(a));
  const floors = positionalFloors(playerPool);
  const premium = benchPremium(bench, floors);
  const premiumLimit = Number(options.benchPremiumLimit ?? BENCH_PREMIUM_LIMIT);
  if (!options.benchBoost && premium > premiumLimit + 1e-6) return null;

  const outfieldStarters = lineup.xi.filter((player) => player.position !== "GK");
  const missDistribution = absenceDistribution(outfieldStarters);
  const autosubPoints = outfieldBench.reduce(
    (sum, player, index) =>
      sum +
      Math.min(
        probabilityAtLeast(missDistribution, index + 1),
        [0.30, 0.09, 0.025][index],
      ) *
        Number(player.projected) *
        0.82,
    0,
  );
  const startingGoalkeeper = lineup.xi.find((player) => player.position === "GK");
  const goalkeeperAutosub =
    Number(benchGoalkeeper?.projected ?? 0) *
    Math.min(
      0.10,
      Math.max(
        0,
        1 - Number(startingGoalkeeper?.minutesModel?.playProbability ?? 100) / 100,
      ),
    );
  const rotationOption = options.benchBoost
    ? bench.reduce((sum, player) => sum + immediateUtility(player), 0)
    : outfieldBench.reduce(
        (sum, player) => sum + 0.035 * Number(player.sixWeekProjected) / 6,
        0,
      );
  const score =
    lineup.utility +
    autosubPoints +
    goalkeeperAutosub +
    rotationOption -
    0.22 * premium;

  return {
    score,
    squad,
    xi: lineup.xi,
    captain: lineup.captain,
    bench: [benchGoalkeeper, ...outfieldBench].filter(Boolean),
    benchPremium: premium,
    benchSpend: bench.reduce((sum, player) => sum + Number(player.price), 0),
    spend,
    autosubPoints: autosubPoints + goalkeeperAutosub,
  };
}

function candidateShortlist(players) {
  const chosen = new Map();
  const add = (rows) => rows.forEach((player) => chosen.set(player.id, player));
  for (const position of Object.keys(POSITION_QUOTA)) {
    const pool = players.filter((player) => player.position === position);
    add([...pool].sort((a, b) => immediateUtility(b) - immediateUtility(a)).slice(0, 14));
    add([...pool].sort((a, b) => captainUtility(b) - captainUtility(a)).slice(0, 8));
    add(
      [...pool]
        .sort(
          (a, b) =>
            immediateUtility(b) / Number(b.price) - immediateUtility(a) / Number(a.price),
        )
        .slice(0, 14),
    );
    add(
      [...pool]
        .sort((a, b) => Number(b.sixWeekProjected) - Number(a.sixWeekProjected))
        .slice(0, 8),
    );
    add(
      [...pool]
        .filter((player) => Number(player.minutesModel?.playProbability ?? 0) >= 70)
        .sort((a, b) => Number(a.price) - Number(b.price))
        .slice(0, 8),
    );
    add([...pool].sort((a, b) => Number(a.price) - Number(b.price)).slice(0, 6));
  }
  return [...chosen.values()];
}

function greedySeed(players, pricePenalty, formation, floors, premiumLimit) {
  const selected = [];
  const positions = {};
  const teams = {};
  const ordered = [...players].sort((a, b) => {
    const aSeed =
      immediateUtility(a) + 0.14 * captainUtility(a) - pricePenalty * Number(a.price);
    const bSeed =
      immediateUtility(b) + 0.14 * captainUtility(b) - pricePenalty * Number(b.price);
    return bSeed - aSeed;
  });
  for (const player of ordered) {
    if ((positions[player.position] ?? 0) >= formation[player.position]) continue;
    if ((teams[player.team] ?? 0) >= 3) continue;
    selected.push(player);
    positions[player.position] = (positions[player.position] ?? 0) + 1;
    teams[player.team] = (teams[player.team] ?? 0) + 1;
    if (selected.length === 11) break;
  }
  if (selected.length !== 11) return [];

  let premium = 0;
  const benchOrdered = [...players].sort((a, b) => {
    const aBench =
      0.18 * immediateUtility(a) +
      0.04 * Number(a.sixWeekProjected) / 6 -
      (0.18 + 0.35 * pricePenalty) * Number(a.price);
    const bBench =
      0.18 * immediateUtility(b) +
      0.04 * Number(b.sixWeekProjected) / 6 -
      (0.18 + 0.35 * pricePenalty) * Number(b.price);
    return bBench - aBench;
  });
  for (const player of benchOrdered) {
    if (selected.some((candidate) => candidate.id === player.id)) continue;
    if ((positions[player.position] ?? 0) >= POSITION_QUOTA[player.position]) continue;
    if ((teams[player.team] ?? 0) >= 3) continue;
    const playerPremium = Math.max(0, Number(player.price) - floors[player.position]);
    if (premium + playerPremium > premiumLimit + 1e-6) continue;
    selected.push(player);
    premium += playerPremium;
    positions[player.position] = (positions[player.position] ?? 0) + 1;
    teams[player.team] = (teams[player.team] ?? 0) + 1;
    if (selected.length === 15) break;
  }
  return selected;
}

function improve(result, candidates, playerPool, options) {
  let best = result;
  for (let pass = 0; pass < 2; pass += 1) {
    let improved = best;
    const currentIds = new Set(best.squad.map((player) => player.id));
    for (let index = 0; index < best.squad.length; index += 1) {
      const outgoing = best.squad[index];
      for (const incoming of candidates) {
        if (incoming.position !== outgoing.position || currentIds.has(incoming.id)) continue;
        const trial = [...best.squad];
        trial[index] = incoming;
        const evaluated = evaluateSquad(trial, playerPool, options);
        if (evaluated && evaluated.score > improved.score + 1e-8) improved = evaluated;
      }
    }
    if (improved === best) break;
    best = improved;
  }
  return best;
}

export function buildOptimizedSquad(players, options = {}) {
  const candidates = candidateShortlist(players);
  const floors = positionalFloors(players);
  const premiumLimit = Number(options.benchPremiumLimit ?? BENCH_PREMIUM_LIMIT);
  // Let the beam traverse lower-spend intermediate squads. The final gate still
  // requires the fresh squad to use at least £99.5m.
  const seedOptions = { ...options, allowStrategicBank: true };
  const seeds = new Map();
  for (const formation of FORMATIONS) {
    for (let step = 0; step <= 60; step += 1) {
      const squad = greedySeed(players, step * 0.02, formation, floors, premiumLimit);
      const evaluated = evaluateSquad(squad, players, seedOptions);
      if (!evaluated) continue;
      const key = squad.map((player) => player.id).sort((a, b) => a - b).join("-");
      const existing = seeds.get(key);
      if (!existing || evaluated.score > existing.score) seeds.set(key, evaluated);
    }
  }

  const finalists = [...seeds.values()].sort((a, b) => b.score - a.score).slice(0, 2);
  if (finalists.length === 0) return null;
  const compliant = finalists
    .map((result) => improve(result, candidates, players, seedOptions))
    .map((result) => evaluateSquad(result.squad, players, options))
    .filter(Boolean)
    .map((result) => improve(result, candidates, players, options));
  return compliant.sort((a, b) => b.score - a.score)[0] ?? null;
}

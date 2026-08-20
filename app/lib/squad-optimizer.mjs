import { solve } from "yalps";

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
export const MIN_XI_START_PROBABILITY = 70;
export const MIN_XI_PLAY_PROBABILITY = 84;

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

function benchUtility(player) {
  const playProbability = Number(player.minutesModel?.playProbability ?? 100) / 100;
  return (
    0.045 * Number(player.sixWeekProjected) / 6 +
    0.055 * playProbability * Math.min(Number(player.projected), 4.5)
  );
}

function exceptionalUpside(player, players) {
  const startProbability = Number(player.minutesModel?.startProbability ?? 0);
  const playProbability = Number(player.minutesModel?.playProbability ?? 0);
  const immediate = Number(player.projected);
  const sorted = players.map((candidate) => Number(candidate.projected)).sort((a, b) => a - b);
  const threshold = sorted[Math.max(0, Math.floor(0.95 * (sorted.length - 1)))] ?? Infinity;
  return (
    startProbability >= 70 &&
    playProbability >= 78 &&
    immediate >= threshold
  );
}

function isStandardStarter(player) {
  const startProbability = Number(player.minutesModel?.startProbability ?? 0);
  const playProbability = Number(player.minutesModel?.playProbability ?? 0);
  return (
    startProbability >= MIN_XI_START_PROBABILITY &&
    playProbability >= MIN_XI_PLAY_PROBABILITY
  );
}

function requiresException(player, players) {
  return !isStandardStarter(player) && exceptionalUpside(player, players);
}

function isSafeStarter(player, players) {
  return isStandardStarter(player) || requiresException(player, players);
}

function presolvePlayers(players) {
  const retained = new Set();
  const frontierSize = 8;
  const comparators = [
    (a, b) => immediateUtility(b) - immediateUtility(a),
    (a, b) => captainUtility(b) - captainUtility(a),
    (a, b) => Number(b.sixWeekProjected) - Number(a.sixWeekProjected),
    (a, b) => Number(b.liveScore) - Number(a.liveScore),
    (a, b) =>
      immediateUtility(b) / Math.max(Number(b.price), 0.1) -
      immediateUtility(a) / Math.max(Number(a.price), 0.1),
    (a, b) => {
      const playDifference =
        Number(b.minutesModel?.playProbability ?? 0) -
        Number(a.minutesModel?.playProbability ?? 0);
      return playDifference || Number(a.price) - Number(b.price);
    },
    (a, b) => Number(a.price) - Number(b.price),
  ];
  for (const position of Object.keys(POSITION_QUOTA)) {
    const positionPlayers = players.filter((player) => player.position === position);
    for (const comparator of comparators) {
      positionPlayers
        .toSorted(comparator)
        .slice(0, frontierSize)
        .forEach((player) => retained.add(player.id));
    }
  }
  return players.filter((player) => retained.has(player.id));
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

function chooseLineup(squad, playerPool = squad) {
  let best = null;
  for (const formation of FORMATIONS) {
    const exceptionOptions = [
      null,
      ...squad.filter((player) => requiresException(player, playerPool)),
    ];
    for (const exception of exceptionOptions) {
      const xi = [];
      let valid = true;
      for (const [position, count] of Object.entries(formation)) {
        const exceptionSlots = exception?.position === position ? 1 : 0;
        const standard = squad
          .filter(
            (player) =>
              player.position === position &&
              player.id !== exception?.id &&
              isStandardStarter(player),
          )
          .sort((a, b) => immediateUtility(b) - immediateUtility(a))
          .slice(0, count - exceptionSlots);
        if (standard.length !== count - exceptionSlots) {
          valid = false;
          break;
        }
        xi.push(...standard);
        if (exceptionSlots) xi.push(exception);
      }
      if (!valid || xi.length !== 11) continue;
      const captain = [...xi].sort((a, b) => captainUtility(b) - captainUtility(a))[0];
      const utility =
        xi.reduce((sum, player) => sum + immediateUtility(player), 0) +
        captainUtility(captain);
      if (!best || utility > best.utility) best = { xi, captain, utility };
    }
  }
  return best;
}

export function evaluateSquad(squad, playerPool, options = {}) {
  if (!isLegalSquad(squad)) return null;
  const spend = squad.reduce((sum, player) => sum + Number(player.price), 0);
  const minimumSpend = Number(options.minimumSpend ?? 99.5);
  if (!options.allowStrategicBank && spend < minimumSpend - 1e-6) return null;
  const lineup = chooseLineup(squad, playerPool);
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
    : bench.reduce((sum, player) => sum + benchUtility(player), 0);
  const score =
    lineup.utility +
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

export function buildOptimizedSquad(players, options = {}) {
  if (!players.length) return null;
  const referencePlayers = players;
  const floors = positionalFloors(referencePlayers);
  players = presolvePlayers(referencePlayers);
  const premiumLimit = Number(options.benchPremiumLimit ?? BENCH_PREMIUM_LIMIT);
  const minimumSpend = options.allowStrategicBank
    ? Number(options.minimumSpend ?? 0)
    : Number(options.minimumSpend ?? 99.5);
  const constraints = {
    budget: { min: minimumSpend, max: 100 },
    squad_total: { equal: 15 },
    xi_total: { equal: 11 },
    captain_total: { equal: 1 },
    bench_premium: options.benchBoost ? { min: 0 } : { max: premiumLimit },
    exception_xi: { max: 1 },
    squad_GK: { equal: 2 },
    squad_DEF: { equal: 5 },
    squad_MID: { equal: 5 },
    squad_FWD: { equal: 3 },
    xi_GK: { equal: 1 },
    xi_DEF: { min: 3, max: 5 },
    xi_MID: { min: 2, max: 5 },
    xi_FWD: { min: 1, max: 3 },
  };
  for (const player of players) {
    constraints[`club_${player.team}`] ??= { max: 3 };
    constraints[`xi_link_${player.id}`] = { max: 0 };
    constraints[`captain_link_${player.id}`] = { max: 0 };
    if (!isSafeStarter(player, referencePlayers)) constraints[`xi_allowed_${player.id}`] = { max: 0 };
  }

  const variables = {};
  for (const player of players) {
    const id = String(player.id);
    const premium = Math.max(0, Number(player.price) - floors[player.position]);
    const bench = options.benchBoost ? immediateUtility(player) : benchUtility(player);
    variables[`s_${id}`] = {
      objective: bench - 0.22 * premium,
      budget: Number(player.price),
      squad_total: 1,
      [`squad_${player.position}`]: 1,
      [`club_${player.team}`]: 1,
      bench_premium: premium,
      [`xi_link_${id}`]: -1,
    };
    variables[`x_${id}`] = {
      objective: immediateUtility(player) - bench + 0.22 * premium,
      xi_total: 1,
      [`xi_${player.position}`]: 1,
      bench_premium: -premium,
      exception_xi: requiresException(player, referencePlayers) ? 1 : 0,
      [`xi_link_${id}`]: 1,
      [`captain_link_${id}`]: -1,
      ...(!isSafeStarter(player, referencePlayers) ? { [`xi_allowed_${id}`]: 1 } : {}),
    };
    variables[`c_${id}`] = {
      objective: captainUtility(player),
      captain_total: 1,
      [`captain_link_${id}`]: 1,
    };
  }
  const solution = solve(
    {
      direction: "maximize",
      objective: "objective",
      constraints,
      variables,
      binaries: true,
    },
    {
      tolerance: 0,
      timeout: Number(options.timeoutMs ?? 12_000),
      maxIterations: 100_000,
      maxPivots: 100_000,
    },
  );
  if (solution.status !== "optimal") {
    throw new Error(`Exact squad MILP failed with status ${solution.status}`);
  }
  const active = new Set(
    solution.variables.filter(([, value]) => value > 0.5).map(([key]) => key),
  );
  const squad = players.filter((player) => active.has(`s_${player.id}`));
  const xi = players.filter((player) => active.has(`x_${player.id}`));
  const captain = players.find((player) => active.has(`c_${player.id}`));
  const xiIds = new Set(xi.map((player) => player.id));
  const bench = squad
    .filter((player) => !xiIds.has(player.id))
    .sort((a, b) => {
      if (a.position === "GK") return -1;
      if (b.position === "GK") return 1;
      return immediateUtility(b) - immediateUtility(a);
    });
  const spend = squad.reduce((sum, player) => sum + Number(player.price), 0);
  const premium = benchPremium(bench, floors);
  if (
    squad.length !== 15 ||
    xi.length !== 11 ||
    !captain ||
    !xiIds.has(captain.id) ||
    !isLegalSquad(squad) ||
    (!options.allowStrategicBank && spend < minimumSpend - 1e-6) ||
    (!options.benchBoost && premium > premiumLimit + 1e-6) ||
    xi.filter((player) => requiresException(player, referencePlayers)).length > 1 ||
    xi.some((player) => !isSafeStarter(player, referencePlayers))
  ) {
    throw new Error("Exact squad MILP returned an invalid squad or XI");
  }
  return {
    squad,
    xi,
    captain,
    bench,
    benchPremium: premium,
    benchSpend: bench.reduce((sum, player) => sum + Number(player.price), 0),
    spend,
    autosubPoints: 0,
    score: solution.result,
    solver: {
      type: "exact-binary-milp",
      status: solution.status,
      inputPlayers: referencePlayers.length,
      candidatePlayers: players.length,
      variables: Object.keys(variables).length,
      optimalityGap: 0,
    },
  };
}

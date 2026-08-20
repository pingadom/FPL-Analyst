import results from "../../data/model-results.json";
import currentPlayers from "../../data/current-players.json";

type BootstrapElement = {
  id: number;
  status: string;
  chance_of_playing_next_round: number | null;
  news: string;
  selected_by_percent: string;
};

const STATUS_DEFAULT_CHANCE: Record<string, number> = {
  a: 100,
  d: 75,
  i: 0,
  s: 0,
  u: 0,
  n: 0,
};

function resolvedOfficialChance(player: BootstrapElement) {
  return player.chance_of_playing_next_round
    ?? STATUS_DEFAULT_CHANCE[player.status.toLowerCase()]
    ?? 0;
}

async function deadlineAvailabilityOverlay() {
  const players = structuredClone(currentPlayers);
  try {
    const response = await fetch("https://fantasy.premierleague.com/api/bootstrap-static/", {
      cache: "no-store",
      headers: { "User-Agent": "FPL-Lens/8.0" },
    });
    if (!response.ok) throw new Error(`Official FPL API returned ${response.status}`);
    const bootstrap = (await response.json()) as { elements: BootstrapElement[] };
    const official = new Map(bootstrap.elements.map((player) => [player.id, player]));
    for (const player of players) {
      const latest = official.get(player.id);
      if (!latest) continue;
      const previousChance = Math.max(
        1,
        Number(player.minutesModel.availabilityEvidence.chance ?? 100),
      );
      const chance = Number(resolvedOfficialChance(latest));
      const availabilityRatio = Math.max(0, Math.min(1.25, chance / previousChance));
      const previousProjection = player.projected;
      const previousExpectedMinutes = Math.max(1, player.expectedMinutes);
      player.ownership = Number(latest.selected_by_percent || player.ownership);
      player.minutesModel.startProbability = Math.round(
        Math.min(98, player.minutesModel.startProbability * availabilityRatio),
      );
      player.minutesModel.playProbability = Math.round(
        Math.min(99, player.minutesModel.playProbability * availabilityRatio),
      );
      player.minutesModel.sixtyProbability = Math.round(
        Math.min(98, player.minutesModel.sixtyProbability * availabilityRatio),
      );
      player.expectedMinutes = Math.round(previousExpectedMinutes * availabilityRatio);
      player.minutesModel.scenarios = [
        {
          ...player.minutesModel.scenarios[0],
          probability: player.minutesModel.startProbability,
        },
        {
          ...player.minutesModel.scenarios[1],
          probability: Math.max(
            0,
            player.minutesModel.playProbability - player.minutesModel.startProbability,
          ),
        },
        {
          ...player.minutesModel.scenarios[2],
          probability: Math.max(0, 100 - player.minutesModel.playProbability),
        },
      ];
      const projectionRatio = Math.max(
        0,
        Math.min(1.25, player.expectedMinutes / previousExpectedMinutes),
      );
      player.projected = Number((previousProjection * projectionRatio).toFixed(1));
      player.sixWeekProjected = Number(
        Math.max(0, player.sixWeekProjected - previousProjection + player.projected).toFixed(1),
      );
      player.valueProjected = Number(
        Math.max(0, player.sixWeekProjected / Math.max(0.1, player.price)).toFixed(2),
      );
      player.captainRating = Math.round(player.captainRating * projectionRatio);
      player.score = Math.round(player.score * projectionRatio);
      player.confidence = Math.round(player.confidence * Math.min(1, projectionRatio));
      player.features.minutes = Number(
        Math.max(0, player.features.minutes * projectionRatio).toFixed(4),
      );
      player.researchFeatures.expected_minutes = player.expectedMinutes;
      player.researchFeatures.play_probability = player.minutesModel.playProbability / 100;
      player.researchFeatures.start_probability = player.minutesModel.startProbability / 100;
      player.researchFeatures.sixty_probability = player.minutesModel.sixtyProbability / 100;
      player.researchFeatures.component_xpts = player.projected;
      for (const key of Object.keys(player.components) as Array<keyof typeof player.components>) {
        player.components[key] = Number((player.components[key] * projectionRatio).toFixed(2));
      }
      player.distribution.p10 = Number((player.distribution.p10 * projectionRatio).toFixed(1));
      player.distribution.median = player.projected;
      player.distribution.p90 = Number((player.distribution.p90 * projectionRatio).toFixed(1));
      player.distribution.standardDeviation = Number(
        (player.distribution.standardDeviation * projectionRatio).toFixed(2),
      );
      player.distribution.return5Probability = Math.round(
        Math.min(100, player.distribution.return5Probability * projectionRatio),
      );
      player.distribution.haul8Probability = Math.round(
        Math.min(100, player.distribution.haul8Probability * projectionRatio),
      );
      player.distribution.blankProbability = Math.round(
        Math.max(
          0,
          Math.min(
            100,
            100 - (100 - player.distribution.blankProbability) * projectionRatio,
          ),
        ),
      );
      for (const key of Object.keys(player.strategyScores) as Array<keyof typeof player.strategyScores>) {
        player.strategyScores[key] = Number(
          (player.strategyScores[key] * projectionRatio).toFixed(4),
        );
      }
      player.minutesModel.availabilityEvidence = {
        status: latest.status,
        chance,
        officialNews: latest.news || "No official flag",
      };
      player.riskFlags = player.riskFlags.filter(
        (flag) => flag !== "Fitness flag" && flag !== "Deadline availability changed",
      );
      if (latest.status !== "a" || chance < 100) {
        player.riskFlags.unshift("Deadline availability changed");
      }
    }
    return { players, source: "live-official-fpl", poolCount: bootstrap.elements.length };
  } catch {
    return { players, source: "generated-artifact-fallback", poolCount: players.length };
  }
}

export async function GET(request: Request) {
  const url = new URL(request.url);
  const position = url.searchParams.get("position")?.toUpperCase();
  const requestedLimit = Number(url.searchParams.get("limit") ?? 100);
  const live = await deadlineAvailabilityOverlay();
  const limit = Number.isFinite(requestedLimit)
    ? Math.min(live.players.length, Math.max(1, Math.floor(requestedLimit)))
    : 100;
  const players = live.players
    .filter((player) => !position || player.position === position)
    .slice(0, limit);

  return Response.json(
    {
      product: results.product,
      model: results.model.version,
      generatedAt: results.generatedAt,
      gameweek: results.headline.gameweek,
      deadline: results.headline.deadline,
      methodology: results.model.method,
      availabilitySource: live.source,
      officialPoolCount: live.poolCount,
      count: players.length,
      players,
    },
    {
      headers: {
        "Access-Control-Allow-Origin": "*",
        "Cache-Control": "public, max-age=30, s-maxage=60",
      },
    },
  );
}

export function OPTIONS() {
  return new Response(null, {
    status: 204,
    headers: {
      "Access-Control-Allow-Origin": "*",
      "Access-Control-Allow-Methods": "GET, OPTIONS",
      "Access-Control-Allow-Headers": "Content-Type",
      "Access-Control-Max-Age": "86400",
    },
  });
}

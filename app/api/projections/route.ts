import results from "../../data/model-results.json";
import currentPlayers from "../../data/current-players.json";

type BootstrapElement = {
  id: number;
  status: string;
  chance_of_playing_next_round: number | null;
  news: string;
  selected_by_percent: string;
};

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
      const chance = Number(
        latest.chance_of_playing_next_round ?? (latest.status === "a" ? 100 : previousChance),
      );
      const availabilityRatio = Math.max(0, Math.min(1.25, chance / previousChance));
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

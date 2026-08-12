import results from "../../data/model-results.json";
import currentPlayers from "../../data/current-players.json";

export async function GET(request: Request) {
  const url = new URL(request.url);
  const position = url.searchParams.get("position")?.toUpperCase();
  const requestedLimit = Number(url.searchParams.get("limit") ?? 100);
  const limit = Number.isFinite(requestedLimit)
    ? Math.min(391, Math.max(1, Math.floor(requestedLimit)))
    : 100;
  const players = currentPlayers
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
      count: players.length,
      players,
    },
    {
      headers: {
        "Access-Control-Allow-Origin": "*",
        "Cache-Control": "public, max-age=300, s-maxage=900",
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

import breakthrough from "../../data/breakthrough-v3.json";
import chipScenarios from "../../data/chip-scenarios.json";
import deadlineStatus from "../../data/deadline-status.json";
import frontierScores from "../../data/frontier-scores.json";
import listwiseScores from "../../data/listwise-scores.json";
import performanceProgress from "../../data/performance-progress.json";
import shadowStatus from "../../data/shadow-status.json";

export async function GET() {
  return Response.json(
    {
      deadline: deadlineStatus,
      breakthrough,
      chips: chipScenarios,
      shadows: shadowStatus,
      performance: performanceProgress,
      frontier: {
        status: frontierScores.status,
        model: frontierScores.model,
        historicalBest: frontierScores.historicalBest,
        historicalValidation: frontierScores.historicalValidation,
        promotionRule: frontierScores.promotionRule,
      },
      listwise: {
        status: listwiseScores.status,
        model: listwiseScores.model,
        historicalValidation: listwiseScores.historicalValidation,
        captainValidation: listwiseScores.captainValidation,
        promotionRule: listwiseScores.promotionRule,
      },
    },
    {
      headers: {
        "Access-Control-Allow-Origin": "*",
        "Cache-Control": "public, max-age=120, s-maxage=300",
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

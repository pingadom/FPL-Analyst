import results from "../../data/model-results.json";
import currentPlayers from "../../data/current-players.json";
import { BOOTSTRAP_TTL_MS, BOOTSTRAP_URL, officialJson } from "../../lib/fpl-api";

type BootstrapTeam = { id: number; name: string; short_name: string };
type BootstrapEvent = { id: number; deadline_time: string; is_next: boolean; finished: boolean };
type OfficialFixture = {
  event: number | null;
  team_h: number;
  team_a: number;
  team_h_difficulty: number;
  team_a_difficulty: number;
  kickoff_time: string | null;
};

const HORIZON = 6;

// The model's own view of each club, taken from any of its players: attack and
// defence ranks are league-wide (1 = strongest).
function modelTeamRatings() {
  const ratings: Record<string, { attack: number; defence: number; strength: number }> = {};
  for (const player of currentPlayers) {
    if (ratings[player.team]) continue;
    ratings[player.team] = {
      attack: player.teamContext.attackRank,
      defence: player.teamContext.defenceRank,
      strength: player.teamContext.strengthRank,
    };
  }
  return ratings;
}

export async function GET() {
  try {
    const [bootstrap, fixtures] = await Promise.all([
      officialJson<{ teams: BootstrapTeam[]; events: BootstrapEvent[] }>(BOOTSTRAP_URL, BOOTSTRAP_TTL_MS),
      officialJson<OfficialFixture[]>(
        "https://fantasy.premierleague.com/api/fixtures/?future=1",
        5 * 60_000,
      ),
    ]);
    const next =
      bootstrap.events.find((event) => event.is_next)?.id ?? results.headline.gameweek;
    const events = Array.from({ length: HORIZON }, (_, index) => next + index).filter(
      (event) => bootstrap.events.some((item) => item.id === event),
    );
    const short = new Map(bootstrap.teams.map((team) => [team.id, team.short_name]));
    const teams = bootstrap.teams.map((team) => {
      const byEvent: Record<number, Array<{ opponent: string; home: boolean; difficulty: number; kickoff: string | null }>> = {};
      for (const event of events) byEvent[event] = [];
      for (const fixture of fixtures) {
        if (fixture.event === null || !(fixture.event in byEvent)) continue;
        if (fixture.team_h === team.id) {
          byEvent[fixture.event].push({
            opponent: short.get(fixture.team_a) ?? "?",
            home: true,
            difficulty: fixture.team_h_difficulty,
            kickoff: fixture.kickoff_time,
          });
        } else if (fixture.team_a === team.id) {
          byEvent[fixture.event].push({
            opponent: short.get(fixture.team_h) ?? "?",
            home: false,
            difficulty: fixture.team_a_difficulty,
            kickoff: fixture.kickoff_time,
          });
        }
      }
      return { id: team.id, name: team.name, short: team.short_name, fixtures: byEvent };
    });
    return Response.json(
      { source: "official-fpl", events, teams, modelRatings: modelTeamRatings() },
      { headers: { "Cache-Control": "public, max-age=300, s-maxage=600" } },
    );
  } catch (error) {
    return Response.json(
      {
        source: "unavailable",
        error: error instanceof Error ? error.message : "Fixture list unavailable",
        events: [],
        teams: [],
        modelRatings: modelTeamRatings(),
      },
      { status: 200, headers: { "Cache-Control": "no-store" } },
    );
  }
}

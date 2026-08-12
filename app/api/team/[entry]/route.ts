import results from "../../../data/model-results.json";
import currentPlayers from "../../../data/current-players.json";

type EntrySummary = {
  id: number;
  name: string;
  player_first_name: string;
  player_last_name: string;
  summary_overall_points: number;
  summary_overall_rank: number;
  current_event?: number;
  last_deadline_bank?: number;
  last_deadline_value?: number;
};

type Pick = {
  element: number;
  position: number;
  multiplier: number;
  purchase_price: number;
  selling_price: number;
};

async function officialJson<T>(url: string): Promise<T> {
  const response = await fetch(url, {
    headers: { "User-Agent": "FPL-Lens/7.0" },
    cache: "no-store",
    signal: AbortSignal.timeout(8_000),
  });
  if (!response.ok) throw new Error(`Official FPL API returned ${response.status}`);
  return response.json() as Promise<T>;
}

export async function GET(
  _request: Request,
  context: { params: Promise<{ entry: string }> },
) {
  const { entry } = await context.params;
  if (!/^\d{1,10}$/.test(entry)) {
    return Response.json({ error: "Enter a valid numeric FPL team ID." }, { status: 400 });
  }

  try {
    const [manager, history, bootstrap] = await Promise.all([
      officialJson<EntrySummary>(`https://fantasy.premierleague.com/api/entry/${entry}/`),
      officialJson<{ current: Array<{ event: number; points: number; total_points: number; overall_rank: number }> }>(
        `https://fantasy.premierleague.com/api/entry/${entry}/history/`,
      ),
      officialJson<{ total_players: number }>(
        "https://fantasy.premierleague.com/api/bootstrap-static/",
      ),
    ]);

    const latestFinished = history.current.at(-1)?.event ?? 0;
    const requestedEvent = Math.max(
      1,
      Math.min(manager.current_event ?? results.headline.gameweek, Math.max(1, latestFinished)),
    );
    let picksEvent = requestedEvent;
    let picks: Pick[] = [];
    for (const event of [requestedEvent, Math.max(1, requestedEvent - 1)]) {
      try {
        const payload = await officialJson<{ picks: Pick[] }>(
          `https://fantasy.premierleague.com/api/entry/${entry}/event/${event}/picks/`,
        );
        picks = payload.picks;
        picksEvent = event;
        break;
      } catch {
        // The upcoming lineup is hidden until its deadline; use the latest visible squad.
      }
    }

    const projectionById = new Map(currentPlayers.map((player) => [player.id, player]));
    const owned = picks
      .map((pick) => {
        const player = projectionById.get(pick.element);
        return player ? { ...pick, player } : null;
      })
      .filter((item): item is NonNullable<typeof item> => item !== null);
    const ownedIds = new Set(owned.map((item) => item.element));
    const bank = (manager.last_deadline_bank ?? 0) / 10;
    const suggestions: Array<{
      sell: (typeof currentPlayers)[number];
      buy: (typeof currentPlayers)[number];
      horizonGain: number;
      affordable: boolean;
    }> = [];
    const targets = currentPlayers
      .filter((player) => !ownedIds.has(player.id))
      .sort((a, b) => b.sixWeekProjected - a.sixWeekProjected);
    const exits = [...owned].sort(
      (a, b) => a.player.sixWeekProjected - b.player.sixWeekProjected,
    );
    const usedTargets = new Set<number>();
    const ownedClubCounts = owned.reduce<Record<string, number>>((counts, item) => {
      counts[item.player.team] = (counts[item.player.team] ?? 0) + 1;
      return counts;
    }, {});
    for (const exit of exits) {
      const target = targets.find(
        (candidate) =>
          !usedTargets.has(candidate.id) &&
          candidate.position === exit.player.position &&
          (ownedClubCounts[candidate.team] ?? 0) -
            (candidate.team === exit.player.team ? 1 : 0) < 3 &&
          candidate.price <= exit.selling_price / 10 + bank &&
          candidate.sixWeekProjected > exit.player.sixWeekProjected + 1.5,
      );
      if (!target) continue;
      usedTargets.add(target.id);
      suggestions.push({
        sell: exit.player,
        buy: target,
        horizonGain: Number((target.sixWeekProjected - exit.player.sixWeekProjected).toFixed(1)),
        affordable: true,
      });
      if (suggestions.length === 3) break;
    }

    const formations = [
      { GK: 1, DEF: 3, MID: 5, FWD: 2 },
      { GK: 1, DEF: 3, MID: 4, FWD: 3 },
      { GK: 1, DEF: 4, MID: 5, FWD: 1 },
      { GK: 1, DEF: 4, MID: 4, FWD: 2 },
      { GK: 1, DEF: 4, MID: 3, FWD: 3 },
      { GK: 1, DEF: 5, MID: 4, FWD: 1 },
      { GK: 1, DEF: 5, MID: 3, FWD: 2 },
      { GK: 1, DEF: 5, MID: 2, FWD: 3 },
    ] as const;
    const bestLineup = formations
      .map((formation) => {
        const lineup = (Object.keys(formation) as Array<keyof typeof formation>)
          .flatMap((position) => owned
            .filter((item) => item.player.position === position)
            .sort((a, b) => b.player.projected - a.player.projected)
            .slice(0, formation[position]));
        const captain = [...lineup].sort(
          (a, b) => b.player.projected - a.player.projected,
        )[0];
        const projection = lineup.reduce(
          (sum, item) => sum + item.player.projected,
          captain?.player.projected ?? 0,
        );
        return { lineup, captain, projection };
      })
      .filter((option) => option.lineup.length === 11)
      .sort((a, b) => b.projection - a.projection)[0];
    const teamProjection = bestLineup?.projection ?? 0;
    const modelProjection = results.headline.projected;
    const edge = modelProjection - teamProjection;
    const currentRank = manager.summary_overall_rank || history.current.at(-1)?.overall_rank || 0;
    const totalManagers = bootstrap.total_players || results.currentMeta.managerPopulation || 1;
    const currentPercentile = currentRank > 0 ? (100 * currentRank) / totalManagers : 100;
    // Without a projected field-score distribution, converting a single team
    // projection into an exact future rank is false precision. Keep the exact
    // current rank as the anchor and expose an uncertainty band only.
    const projectedMedianRank = currentRank;
    const spread = Math.max(0.12, (results.headline.scenario.p90 - results.headline.scenario.p10) / 100);

    return Response.json(
      {
        manager: {
          id: manager.id,
          teamName: manager.name,
          playerName: `${manager.player_first_name} ${manager.player_last_name}`.trim(),
          points: manager.summary_overall_points,
          overallRank: currentRank,
          totalManagers,
          percentile: Number(currentPercentile.toFixed(2)),
          squadValue: (manager.last_deadline_value ?? 0) / 10,
          bank,
        },
        picksEvent,
        owned,
        suggestions,
        forecast: {
          teamProjection: Number(teamProjection.toFixed(1)),
          modelProjection,
          edge: Number(edge.toFixed(1)),
          medianRank: projectedMedianRank,
          optimisticRank: projectedMedianRank
            ? Math.max(1, Math.round(projectedMedianRank * (1 - spread)))
            : 0,
          cautiousRank: projectedMedianRank
            ? Math.min(totalManagers, Math.round(projectedMedianRank * (1 + spread)))
            : 0,
          method: "The exact current rank is the anchor. The band reflects squad-score uncertainty only; a future-rank forecast is withheld until a calibrated field-score model is available.",
          lineup: bestLineup?.lineup.map((item) => item.element) ?? [],
          captain: bestLineup?.captain?.element ?? null,
        },
      },
      { headers: { "Cache-Control": "private, max-age=60" } },
    );
  } catch (error) {
    return Response.json(
      {
        error:
          error instanceof Error
            ? `Could not load that FPL team: ${error.message}`
            : "Could not load that FPL team.",
      },
      { status: 502 },
    );
  }
}

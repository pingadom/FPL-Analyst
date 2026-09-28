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

type HistoryRow = {
  event: number;
  points: number;
  total_points: number;
  overall_rank: number;
  event_transfers?: number;
};

// Free transfers are not in the public API, so they are rebuilt from the
// manager's own history under the 2024/25+ rules: one more after every
// Gameweek, at most five, and a Wildcard or Free Hit week preserves them.
function estimateFreeTransfers(
  rows: HistoryRow[],
  chips: Array<{ name: string; event: number }>,
): number {
  const chipWeeks = new Set(
    chips
      .filter((chip) => chip.name === "wildcard" || chip.name === "freehit")
      .map((chip) => chip.event),
  );
  const ordered = [...rows].sort((a, b) => a.event - b.event);
  let free = 1;
  for (const row of ordered.slice(1)) {
    const used = chipWeeks.has(row.event) ? 0 : row.event_transfers ?? 0;
    free = Math.min(5, Math.max(0, free - used) + 1);
  }
  return ordered.length ? free : 1;
}

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
      officialJson<{ current: HistoryRow[]; chips?: Array<{ name: string; event: number }> }>(
        `https://fantasy.premierleague.com/api/entry/${entry}/history/`,
      ),
      officialJson<{
        total_players: number;
        elements: Array<{ id: number; web_name: string; team: number; element_type: number; now_cost: number }>;
        teams: Array<{ id: number; short_name: string }>;
      }>(
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
    // Injured, suspended and departed players are not in the projection file,
    // but they are exactly the ones to sell. The replay values a held player
    // who cannot play at -0.30, so they are kept as exits at that value.
    const positionName: Record<number, string> = { 1: "GK", 2: "DEF", 3: "MID", 4: "FWD" };
    const clubName = new Map(bootstrap.teams.map((team) => [team.id, team.short_name]));
    const officialById = new Map(bootstrap.elements.map((element) => [element.id, element]));
    const unavailable = picks
      .filter((pick) => !projectionById.has(pick.element))
      .map((pick) => {
        const official = officialById.get(pick.element);
        if (!official) return null;
        const player = {
          id: official.id,
          name: official.web_name,
          team: clubName.get(official.team) ?? "",
          position: positionName[official.element_type] ?? "",
          price: official.now_cost / 10,
          projected: 0,
          sixWeekProjected: -0.3,
        } as unknown as (typeof currentPlayers)[number];
        return { ...pick, player };
      })
      .filter((item): item is NonNullable<typeof item> => item !== null);
    const bank = (manager.last_deadline_bank ?? 0) / 10;
    // The same transfer rule the model's replay was validated with: take the
    // single best affordable same-position swap by six-week gain, up to the free
    // transfers available, and only while it clears the published hurdle. That
    // hurdle is large on purpose -- only about 37% of a predicted transfer gain
    // is realised -- so marginal swaps are banked rather than suggested.
    const hurdle =
      (results.headline as { transferHurdle?: number }).transferHurdle ?? 4.3;
    const freeTransfers = estimateFreeTransfers(history.current, history.chips ?? []);
    const suggestions: Array<{
      sell: (typeof currentPlayers)[number];
      buy: (typeof currentPlayers)[number];
      horizonGain: number;
      affordable: boolean;
    }> = [];
    let budget = bank;
    const squad = [...owned, ...unavailable];
    const held = new Map(squad.map((item) => [item.element, item]));
    const clubCounts = squad.reduce<Record<string, number>>((counts, item) => {
      counts[item.player.team] = (counts[item.player.team] ?? 0) + 1;
      return counts;
    }, {});
    for (let move = 0; move < Math.min(freeTransfers, 5); move += 1) {
      let best: { exitId: number; target: (typeof currentPlayers)[number]; gain: number } | null = null;
      for (const exit of held.values()) {
        for (const target of currentPlayers) {
          if (held.has(target.id) || target.position !== exit.player.position) continue;
          if (target.price > exit.selling_price / 10 + budget) continue;
          const sameClub = target.team === exit.player.team;
          if (!sameClub && (clubCounts[target.team] ?? 0) >= 3) continue;
          const gain = target.sixWeekProjected - exit.player.sixWeekProjected;
          if (!best || gain > best.gain) best = { exitId: exit.element, target, gain };
        }
      }
      if (!best || best.gain <= hurdle) break;
      const exit = held.get(best.exitId);
      if (!exit) break;
      budget += exit.selling_price / 10 - best.target.price;
      clubCounts[exit.player.team] -= 1;
      clubCounts[best.target.team] = (clubCounts[best.target.team] ?? 0) + 1;
      held.delete(best.exitId);
      held.set(best.target.id, {
        ...exit,
        element: best.target.id,
        selling_price: Math.round(best.target.price * 10),
        player: best.target,
      });
      suggestions.push({
        sell: exit.player,
        buy: best.target,
        horizonGain: Number(best.gain.toFixed(1)),
        affordable: true,
      });
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
        transferPolicy: {
          hurdle,
          freeTransfers,
          note: "Free transfers are estimated from your public history; FPL does not publish them without a login.",
        },
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

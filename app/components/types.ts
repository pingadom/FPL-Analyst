export type Position = "GK" | "DEF" | "MID" | "FWD";

export const POSITIONS: Position[] = ["GK", "DEF", "MID", "FWD"];

export type Player = {
  id: number;
  name: string;
  team: string;
  position: Position;
  price: number;
  ownership: number;
  projected: number;
  sixWeekProjected: number;
  riskAdjustedHorizonProjected: number;
  horizonWeightedGames: number;
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
    scenarios: Array<{ label: string; probability: number; minutes: number }>;
    availabilityEvidence: { status: string; chance: number; officialNews: string };
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
  };
  marketForecast: { priceRiseProbability: number; priceFallProbability: number };
  components: {
    appearance: number;
    goals: number;
    assists: number;
    cleanSheet: number;
    defence: number;
    bonus: number;
    adjustment: number;
  };
  history: { matches: number; average: number; per90: number; returnRate: number; volatility: number };
  opponentHistory: { matches: number; average: number; per90: number; returnRate: number };
  teamContext: {
    expectedGoalsFor: number;
    expectedGoalsAgainst: number;
    cleanSheetProbability: number;
    horizonCleanSheetProbability: number;
    attackRank: number;
    defenceRank: number;
    strengthRank: number;
    modelWinProbability: number;
    marketWinProbability: number;
  };
  comparison: {
    positionRank: number;
    positionPlayers: number;
    popularRival: string;
    popularRivalOwnership: number;
    popularRivalProjection: number;
    edgeVsPopular: number;
  };
  optimization: {
    lineupUtility: Record<"protect" | "balanced" | "chase", number>;
    benchUtility: number;
    captainUtility: number;
    standardStarterEligible: boolean;
    exceptionalStarterEligible: boolean;
  };
  strategyScores: Record<"protect" | "balanced" | "chase", number>;
  opponent: string;
  venue: string;
  starter: boolean;
  captain: boolean;
  vice: boolean;
  purchasePrice?: number;
  trend: "up" | "down" | "flat";
};

export type Pick = {
  element: number;
  multiplier: number;
  selling_price: number;
  player: Player;
};

export type ImportedTeam = {
  manager: {
    id: number;
    teamName: string;
    playerName: string;
    points: number;
    overallRank: number;
    totalManagers: number;
    percentile: number;
    squadValue: number;
    bank: number;
  };
  picksEvent: number;
  owned: Pick[];
  unavailable?: Array<{
    element: number;
    selling_price: number;
    name: string;
    team: string;
    position: Position;
    price: number;
  }>;
  history?: Array<{ event: number; points: number; totalPoints: number; overallRank: number; transfers: number }>;
  chipsUsed?: Array<{ name: string; event: number }>;
  suggestions: Array<{ sell: Player; buy: Player; horizonGain: number }>;
  transferPolicy?: { hurdle: number; freeTransfers: number; note: string };
  forecast: { teamProjection: number; modelProjection: number; edge: number };
};

export type FixtureCell = { opponent: string; home: boolean; difficulty: number; kickoff: string | null };

export type FixturesPayload = {
  source: string;
  error?: string;
  events: number[];
  teams: Array<{ id: number; name: string; short: string; fixtures: Record<string, FixtureCell[]> }>;
  modelRatings: Record<string, { attack: number; defence: number; strength: number }>;
};

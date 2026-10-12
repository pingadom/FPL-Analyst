import type { ReactNode } from "react";
import { availabilityIssue } from "./format";
import type { Player, Position } from "./types";

export function PositionTag({ position }: { position: Position }) {
  return <span className={`pos pos-${position.toLowerCase()}`}>{position}</span>;
}

export function PlayerLink({
  player,
  onOpen,
}: {
  player: Player;
  onOpen: (player: Player) => void;
}) {
  return (
    <button type="button" className="player-link" onClick={() => onOpen(player)}>
      {player.name}
    </button>
  );
}

export function Flag({ player }: { player: Player }) {
  const issue = availabilityIssue(player);
  if (!issue) return null;
  const severe = player.minutesModel.availabilityEvidence.chance <= 50;
  return (
    <span className={`flag ${severe ? "flag-red" : "flag-amber"}`} title={issue}>
      {severe ? "!" : "?"}
    </span>
  );
}

export function Stat({ label, value, sub }: { label: string; value: ReactNode; sub?: ReactNode }) {
  return (
    <div className="stat">
      <span className="stat-label">{label}</span>
      <strong className="stat-value">{value}</strong>
      {sub !== undefined && <span className="stat-sub">{sub}</span>}
    </div>
  );
}

export function Card({
  title,
  aside,
  children,
  className = "",
}: {
  title?: ReactNode;
  aside?: ReactNode;
  children: ReactNode;
  className?: string;
}) {
  return (
    <section className={`card ${className}`}>
      {(title || aside) && (
        <header className="card-head">
          {title && <h2>{title}</h2>}
          {aside && <div className="card-aside">{aside}</div>}
        </header>
      )}
      {children}
    </section>
  );
}

export function Empty({ children }: { children: ReactNode }) {
  return <p className="empty">{children}</p>;
}

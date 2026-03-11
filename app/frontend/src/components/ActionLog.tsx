import { useEffect, useRef } from "react";
import type { ActionLogEntry } from "../types";

interface ActionLogProps {
  entries: ActionLogEntry[];
  playbackActive?: boolean;
}

export default function ActionLog({ entries, playbackActive }: ActionLogProps) {
  const endRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to bottom when new entries appear
  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [entries.length]);

  return (
    <div className="action-log">
      <div className="action-log-header">Action Log</div>
      {entries.length === 0 ? (
        <div className="action-log-entry">Game started...</div>
      ) : (
        entries.map((entry, i) => {
          const isLatest = i === entries.length - 1;
          let cls = "action-log-entry";
          if (isLatest && playbackActive) cls += " action-highlight";
          return (
            <div key={i} className={cls}>
              {entry.description}
            </div>
          );
        })
      )}
      <div ref={endRef} />
    </div>
  );
}

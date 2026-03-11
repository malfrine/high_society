import { useState, useCallback, useEffect, useRef } from "react";
import type { GameResponse, ActionLogEntry } from "../types";
import { fetchNewGame, submitAction } from "../api";

const STORAGE_KEY = "high-society-state";
const PLAYBACK_DELAY_MS = 2000;

interface StoredState {
  response: GameResponse;
  robotType: string;
  playerName: string;
  cumulativeLog: ActionLogEntry[];
}

function loadState(): StoredState | null {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return null;
    return JSON.parse(raw);
  } catch {
    return null;
  }
}

function saveState(state: StoredState) {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
}

function clearState() {
  localStorage.removeItem(STORAGE_KEY);
}

export type GamePhase = "lobby" | "playing" | "game_over";

export function useGame() {
  const saved = loadState();

  const [response, setResponse] = useState<GameResponse | null>(
    saved?.response ?? null
  );
  const [robotType, setRobotType] = useState(saved?.robotType ?? "dqn");
  const [playerName, setPlayerName] = useState(saved?.playerName ?? "You");
  const [cumulativeLog, setCumulativeLog] = useState<ActionLogEntry[]>(
    saved?.cumulativeLog ?? []
  );
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Playback state: queue of robot entries to reveal one at a time
  const [playbackQueue, setPlaybackQueue] = useState<ActionLogEntry[]>([]);
  const [playbackActive, setPlaybackActive] = useState(false);
  // The robot entry currently being "shown" (the one that just played)
  const [activeRobotName, setActiveRobotName] = useState<string | null>(null);
  // Pending response to apply after playback finishes
  const pendingResponse = useRef<GameResponse | null>(null);

  const phase: GamePhase = !response
    ? "lobby"
    : response.game_over
      ? "game_over"
      : "playing";

  // Persist to localStorage
  useEffect(() => {
    if (response) {
      saveState({ response, robotType, playerName, cumulativeLog });
    }
  }, [response, robotType, playerName, cumulativeLog]);

  // Playback timer: drip-feed one robot action at a time
  useEffect(() => {
    if (!playbackActive || playbackQueue.length === 0) {
      if (playbackActive) {
        // Queue exhausted — finalize
        setPlaybackActive(false);
        setActiveRobotName(null);
        if (pendingResponse.current) {
          setResponse(pendingResponse.current);
          pendingResponse.current = null;
        }
      }
      return;
    }

    const timer = setTimeout(() => {
      const [next, ...rest] = playbackQueue;
      setActiveRobotName(next.player_name);
      setCumulativeLog((prev) => [...prev, next]);
      setPlaybackQueue(rest);

      if (rest.length === 0) {
        // Last entry — hold highlight briefly then finalize
        setTimeout(() => {
          setPlaybackActive(false);
          setActiveRobotName(null);
          if (pendingResponse.current) {
            setResponse(pendingResponse.current);
            pendingResponse.current = null;
          }
        }, PLAYBACK_DELAY_MS);
      }
    }, PLAYBACK_DELAY_MS);

    return () => clearTimeout(timer);
  }, [playbackActive, playbackQueue]);

  const startGame = useCallback(
    async (numPlayers: number, rType: string, name: string) => {
      setLoading(true);
      setError(null);
      setRobotType(rType);
      setPlayerName(name);
      setCumulativeLog([]);
      setPlaybackQueue([]);
      setPlaybackActive(false);
      setActiveRobotName(null);
      try {
        const resp = await fetchNewGame(numPlayers, rType);
        if (resp.action_log.length > 0) {
          // Robots moved before human — play them back
          pendingResponse.current = resp;
          setPlaybackQueue(resp.action_log);
          setPlaybackActive(true);
        } else {
          setResponse(resp);
        }
      } catch (e) {
        setError(String(e));
      } finally {
        setLoading(false);
      }
    },
    []
  );

  const playAction = useCallback(
    async (action: number) => {
      if (!response || playbackActive) return;
      setLoading(true);
      setError(null);
      try {
        const resp = await submitAction(
          response.game_state,
          response.current_agent_idx,
          action,
          robotType
        );
        // Append human action immediately
        const humanDesc =
          action === 0 ? `${playerName} passed` : `${playerName} added card ${action}`;
        const humanEntry: ActionLogEntry = {
          player_name: playerName,
          action,
          description: humanDesc,
        };
        setCumulativeLog((prev) => [...prev, humanEntry]);

        if (resp.action_log.length > 0) {
          // Queue robot actions for animated playback
          pendingResponse.current = resp;
          setPlaybackQueue(resp.action_log);
          setPlaybackActive(true);
        } else {
          setResponse(resp);
        }
      } catch (e) {
        setError(String(e));
      } finally {
        setLoading(false);
      }
    },
    [response, robotType, playerName, playbackActive]
  );

  const resetGame = useCallback(() => {
    setResponse(null);
    setCumulativeLog([]);
    setPlaybackQueue([]);
    setPlaybackActive(false);
    setActiveRobotName(null);
    pendingResponse.current = null;
    setError(null);
    clearState();
  }, []);

  return {
    phase,
    response,
    loading,
    error,
    playerName,
    robotType,
    cumulativeLog,
    playbackActive,
    activeRobotName,
    startGame,
    playAction,
    resetGame,
  };
}

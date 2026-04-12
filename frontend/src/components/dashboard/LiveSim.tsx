import { useState, useEffect, useRef, useCallback } from "react";

/* ── Types ────────────────────────────────────────────────────────────────── */

type Lane = { id: string; queue: number; wait: number; density: number };

type SimStep = {
  type: "step";
  step: number;
  phase: number;
  in_yellow: boolean;
  time_in_phase: number;
  action: number;
  action_name: string;
  probs: number[] | null;
  q_values: number[] | null;
  value: number | null;
  lanes: Lane[];
  reward: number;
  cars_through: number;
  total_wait: number;
  constraint_penalty: number;
  episode_total_reward: number;
  episode_total_cars: number;
};

export type AgentId = "ppo" | "dqn" | "fixed" | "actuated";

const AGENT_LABELS: Record<AgentId, string> = {
  ppo: "PPO (RL)",
  dqn: "DQN (RL)",
  fixed: "Fixed-Time",
  actuated: "Actuated",
};

const PHASE_LABELS = ["Green EW", "Yellow EW", "Green NS", "Yellow NS"];

/* ── Intersection SVG ─────────────────────────────────────────────────────── */

function lightColor(phase: number, dir: "EW" | "NS"): string {
  if (phase === 0) return dir === "EW" ? "#4caf50" : "#ef5350";
  if (phase === 1) return dir === "EW" ? "#fdd835" : "#ef5350";
  if (phase === 2) return dir === "NS" ? "#4caf50" : "#ef5350";
  if (phase === 3) return dir === "NS" ? "#fdd835" : "#ef5350";
  return "#ef5350";
}

function IntersectionViz({ step }: { step: SimStep | null }) {
  const W = 340;
  const H = 340;
  const cx = W / 2;
  const cy = H / 2;
  const roadW = 42;
  const roadLen = 100;
  const half = 38; // half of intersection box

  const phase = step?.phase ?? 0;
  const lanes = step?.lanes ?? [];
  const MAX_Q = 8; // queue length that fills the road completely

  const qFill = (i: number) =>
    Math.min(((lanes[i]?.queue ?? 0) / MAX_Q) * roadLen, roadLen);

  const wColor = lightColor(phase, "EW");
  const eColor = lightColor(phase, "EW");
  const nColor = lightColor(phase, "NS");
  const sColor = lightColor(phase, "NS");

  const phaseLabel = PHASE_LABELS[phase] ?? `Phase ${phase}`;
  const timeLabel = step ? `${step.time_in_phase.toFixed(1)}s` : "—";

  return (
    <svg width={W} height={H} viewBox={`0 0 ${W} ${H}`} style={{ display: "block", margin: "0 auto" }}>
      {/* Background */}
      <rect width={W} height={H} fill="#edf0e5" rx={16} />

      {/* Roads */}
      <rect x={cx - half - roadLen} y={cy - roadW / 2} width={roadLen} height={roadW} fill="#9e9e9e" rx={2} /> {/* W */}
      <rect x={cx + half}           y={cy - roadW / 2} width={roadLen} height={roadW} fill="#9e9e9e" rx={2} /> {/* E */}
      <rect x={cx - roadW / 2} y={cy - half - roadLen} width={roadW} height={roadLen} fill="#9e9e9e" rx={2} /> {/* N */}
      <rect x={cx - roadW / 2} y={cy + half}           width={roadW} height={roadLen} fill="#9e9e9e" rx={2} /> {/* S */}

      {/* Intersection box */}
      <rect x={cx - half} y={cy - half} width={half * 2} height={half * 2} fill="#757575" />

      {/* Queue bars  (lanes[0]=W, [1]=E, [2]=N, [3]=S) */}
      <rect /* W queue */
        x={cx - half - qFill(0)} y={cy - roadW / 2 + 2}
        width={qFill(0)} height={roadW - 4}
        fill="rgba(225,127,37,0.75)" rx={2}
      />
      <rect /* E queue */
        x={cx + half} y={cy - roadW / 2 + 2}
        width={qFill(1)} height={roadW - 4}
        fill="rgba(225,127,37,0.75)" rx={2}
      />
      <rect /* N queue */
        x={cx - roadW / 2 + 2} y={cy - half - qFill(2)}
        width={roadW - 4} height={qFill(2)}
        fill="rgba(225,127,37,0.75)" rx={2}
      />
      <rect /* S queue */
        x={cx - roadW / 2 + 2} y={cy + half}
        width={roadW - 4} height={qFill(3)}
        fill="rgba(225,127,37,0.75)" rx={2}
      />

      {/* Traffic light circles */}
      <circle cx={cx - half - roadLen + 16} cy={cy}           r={11} fill={wColor} stroke="white" strokeWidth={2} />
      <circle cx={cx + half + roadLen - 16} cy={cy}           r={11} fill={eColor} stroke="white" strokeWidth={2} />
      <circle cx={cx}                       cy={cy - half - roadLen + 16} r={11} fill={nColor} stroke="white" strokeWidth={2} />
      <circle cx={cx}                       cy={cy + half + roadLen - 16} r={11} fill={sColor} stroke="white" strokeWidth={2} />

      {/* Direction labels */}
      <text x={cx - half - roadLen + 16} y={cy - 18} textAnchor="middle" fontSize={11} fontWeight="700" fill="#1f2618">W</text>
      <text x={cx + half + roadLen - 16} y={cy - 18} textAnchor="middle" fontSize={11} fontWeight="700" fill="#1f2618">E</text>
      <text x={cx + roadW / 2 + 8}       y={cy - half - roadLen + 16} textAnchor="start" fontSize={11} fontWeight="700" fill="#1f2618">N</text>
      <text x={cx + roadW / 2 + 8}       y={cy + half + roadLen - 16} textAnchor="start" fontSize={11} fontWeight="700" fill="#1f2618">S</text>

      {/* Queue counts under each road */}
      {lanes[0] && <text x={cx - half - roadLen / 2} y={cy + roadW / 2 + 14} textAnchor="middle" fontSize={10} fill="#48533c">{lanes[0].queue.toFixed(1)}</text>}
      {lanes[1] && <text x={cx + half + roadLen / 2} y={cy + roadW / 2 + 14} textAnchor="middle" fontSize={10} fill="#48533c">{lanes[1].queue.toFixed(1)}</text>}
      {lanes[2] && <text x={cx + roadW / 2 + 4}      y={cy - half - roadLen / 2 + 4} textAnchor="start" fontSize={10} fill="#48533c">{lanes[2].queue.toFixed(1)}</text>}
      {lanes[3] && <text x={cx + roadW / 2 + 4}      y={cy + half + roadLen / 2 + 4} textAnchor="start" fontSize={10} fill="#48533c">{lanes[3].queue.toFixed(1)}</text>}

      {/* Center label: phase + time */}
      <text x={cx} y={cy - 7} textAnchor="middle" fontSize={10} fontWeight="700" fill="white">{phaseLabel}</text>
      <text x={cx} y={cy + 8} textAnchor="middle" fontSize={10} fill="rgba(255,255,255,0.85)">{timeLabel}</text>
    </svg>
  );
}

/* ── Agent decision panel ─────────────────────────────────────────────────── */

function ProbBar({ label, value, color }: { label: string; value: number; color: string }) {
  return (
    <div className="prob-row">
      <span className="prob-label">{label}</span>
      <div className="prob-track">
        <div className="prob-fill" style={{ width: `${value * 100}%`, background: color }} />
      </div>
      <span className="prob-pct">{(value * 100).toFixed(0)}%</span>
    </div>
  );
}

function AgentDecision({ step, agentType }: { step: SimStep | null; agentType: AgentId }) {
  if (!step) {
    return (
      <div className="decision-panel decision-empty">
        <p>Start the simulation to see agent decisions</p>
      </div>
    );
  }

  const isSwitching = step.action === 1;

  return (
    <div className="decision-panel">
      <div className="decision-header">
        <span className="decision-step">Step {step.step}</span>
        <span className={`decision-action ${isSwitching ? "action-switch" : "action-keep"}`}>
          {step.action_name}
        </span>
      </div>

      {/* PPO: probability bars */}
      {agentType === "ppo" && step.probs && (
        <div className="decision-section">
          <div className="section-title">Action Probabilities</div>
          <ProbBar label="Keep" value={step.probs[0]} color="#4caf50" />
          <ProbBar label="Switch" value={step.probs[1]} color="#e17f25" />
          {step.value !== null && (
            <div className="decision-value">
              Critic value: <strong>{step.value.toFixed(2)}</strong>
            </div>
          )}
        </div>
      )}

      {/* DQN: Q-values */}
      {agentType === "dqn" && step.q_values && (
        <div className="decision-section">
          <div className="section-title">Q-Values</div>
          <div className="qval-row">
            <span className={step.action === 0 ? "qval-chosen" : ""}>Keep: {step.q_values[0].toFixed(3)}</span>
            <span className={step.action === 1 ? "qval-chosen" : ""}>Switch: {step.q_values[1].toFixed(3)}</span>
          </div>
        </div>
      )}

      {/* Baselines: plain reason */}
      {(agentType === "fixed" || agentType === "actuated") && (
        <div className="decision-section">
          <div className="section-title">
            {agentType === "fixed" ? "Fixed schedule" : "Demand sensing"}
          </div>
          <p className="baseline-reason">
            {agentType === "fixed"
              ? `${step.time_in_phase.toFixed(1)}s elapsed in phase`
              : `Phase time: ${step.time_in_phase.toFixed(1)}s`}
          </p>
        </div>
      )}

    </div>
  );
}

/* ── Stats bar ────────────────────────────────────────────────────────────── */

function StatsBar({ step }: { step: SimStep }) {
  return (
    <div className="stats-bar">
      <div className="stat-item">
        <span className="stat-label">Episode Reward</span>
        <span className="stat-value">{step.episode_total_reward.toFixed(1)}</span>
      </div>
      <div className="stat-item">
        <span className="stat-label">Cars Through</span>
        <span className="stat-value">{step.episode_total_cars}</span>
      </div>
      <div className="stat-item">
        <span className="stat-label">Current Wait</span>
        <span className="stat-value">{step.total_wait.toFixed(1)}s</span>
      </div>
      <div className="stat-item">
        <span className="stat-label">Step</span>
        <span className="stat-value">{step.step}</span>
      </div>
    </div>
  );
}

/* ── Main LiveSim component ───────────────────────────────────────────────── */

export default function LiveSim() {
  const [agent, setAgent] = useState<AgentId>("ppo");
  const [running, setRunning] = useState(false);
  const [done, setDone] = useState(false);
  const [currentStep, setCurrentStep] = useState<SimStep | null>(null);
  const [error, setError] = useState<string | null>(null);
  const wsRef = useRef<WebSocket | null>(null);

  const stop = useCallback(() => {
    wsRef.current?.close();
    wsRef.current = null;
    setRunning(false);
  }, []);

  const start = useCallback(() => {
    stop();
    setError(null);
    setCurrentStep(null);
    setDone(false);

    const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
    const url = `${proto}//${window.location.host}/ws/simulate?agent=${agent}&steps=300&seed=4100`;
    const ws = new WebSocket(url);
    wsRef.current = ws;
    setRunning(true);

    ws.onmessage = (e) => {
      const msg = JSON.parse(e.data) as SimStep | { type: "done" } | { type: "error"; message: string };
      if (msg.type === "step") {
        setCurrentStep(msg as SimStep);
      } else if (msg.type === "done") {
        setRunning(false);
        setDone(true);
      } else if (msg.type === "error") {
        setError((msg as { type: "error"; message: string }).message);
        setRunning(false);
      }
    };

    ws.onclose = () => setRunning(false);
    ws.onerror = () => {
      setError(
        "Could not connect to the simulation server. Start it with:\n\n" +
        "  cd traffic-rl\n" +
        "  pip install fastapi uvicorn[standard] websockets\n" +
        "  uvicorn api.server:app --port 8000"
      );
      setRunning(false);
    };
  }, [agent, stop]);

  // Cleanup on unmount
  useEffect(() => () => stop(), [stop]);

  return (
    <div className="livesim">
      {/* Controls */}
      <div className="livesim-controls">
        <select
          value={agent}
          onChange={(e) => { setAgent(e.target.value as AgentId); setCurrentStep(null); setDone(false); }}
          disabled={running}
        >
          {(Object.entries(AGENT_LABELS) as [AgentId, string][]).map(([id, label]) => (
            <option key={id} value={id}>{label}</option>
          ))}
        </select>

        {running ? (
          <button className="stop-btn" onClick={stop}>Stop</button>
        ) : (
          <button onClick={start}>
            {done ? "Run Again" : "Start Simulation"}
          </button>
        )}

        {running && <span className="live-badge">LIVE</span>}
        {done && !running && <span className="done-badge">Done</span>}
      </div>

      {/* Error */}
      {error && (
        <div className="livesim-error">
          <pre>{error}</pre>
        </div>
      )}

      {/* Main layout: viz + decision */}
      <div className="livesim-main">
        <div className="livesim-viz">
          <h3 className="viz-title">Intersection</h3>
          <IntersectionViz step={currentStep} />
          <p className="viz-legend">Orange bars = queued vehicles</p>
        </div>

        <div className="livesim-decision">
          <h3 className="viz-title">Agent Decision</h3>
          <AgentDecision step={currentStep} agentType={agent} />
        </div>
      </div>

      {/* Stats bar */}
      {currentStep && <StatsBar step={currentStep} />}
    </div>
  );
}

import { useState } from "react";
import { Link } from "react-router-dom";
import fixed from "../../data/fixed.json";
import actuated from "../../data/actuated.json";
import ppo from "../../data/ppo.json";
import dqn from "../../data/dqn.json";

import {
  CartesianGrid,
  Label,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import "./Dashboard.css";
import LiveSim from "./LiveSim";

type Controller = "fixed" | "actuated" | "ppo" | "dqn";
type Tab = "results" | "livesim";

type DashboardData = {
  episodes: number[];
  avg_wait: number[];
  throughput: number[];
  queue_length: number[];
  mse: number[];
};

const dataMap = { fixed, actuated, ppo, dqn };

export default function Dashboard() {
  const [tab, setTab] = useState<Tab>("results");
  const [controller, setController] = useState<Controller>("ppo");
  const [data, setData] = useState<DashboardData | null>(null);

  const runDemo = () => {
    setData(dataMap[controller]);
  };

  const handleControllerChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    setController(e.target.value as Controller);
    setData(null);
  };

  return (
    <div className="dashboard-page">
      {/* HEADER */}
      <div className="dashboard-header">
        <div>
          <h1>Traffic RL Dashboard</h1>
          <p className="dashboard-subtitle">
            Compare Fixed, Actuated (baseline), DQN, and PPO agents using precomputed simulation
            results — or watch a live agent control the intersection.
          </p>
        </div>
        <Link className="back-button" to="/">
          Back to Home
        </Link>
      </div>

      {/* TABS */}
      <div className="tab-bar">
        <button
          className={`tab-btn${tab === "results" ? " tab-active" : ""}`}
          onClick={() => setTab("results")}
        >
          Results
        </button>
        <button
          className={`tab-btn${tab === "livesim" ? " tab-active" : ""}`}
          onClick={() => setTab("livesim")}
        >
          Live Sim
        </button>
      </div>

      {/* ── RESULTS TAB ── */}
      {tab === "results" && (
        <>
          {/* CONTROLS */}
          <div className="dashboard-controls">
            <select value={controller} onChange={handleControllerChange}>
              <option value="fixed">Fixed</option>
              <option value="actuated">Actuated</option>
              <option value="ppo">PPO</option>
              <option value="dqn">DQN</option>
            </select>

            <button onClick={runDemo}>Run Demo Scenario</button>
          </div>

          {/* KPIs */}
          {data && (
            <div className="metrics-grid">
              <KPI label="Avg Wait" value={data.avg_wait.at(-1) ?? 0} />
              <KPI label="Throughput" value={data.throughput.at(-1) ?? 0} />
              <KPI label="Queue Length" value={data.queue_length.at(-1) ?? 0} />
              <KPI label="MSE" value={data.mse.at(-1) ?? 0} />
            </div>
          )}

          {/* CHARTS */}
          {data && <Charts data={data} />}
        </>
      )}

      {/* ── LIVE SIM TAB ── */}
      {tab === "livesim" && <LiveSim />}
    </div>
  );
}

/* ---------- KPI ---------- */
function KPI({ label, value }: { label: string; value: number }) {
  return (
    <div className="metric-card">
      <div className="metric-label">{label}</div>
      <div className="metric-value">{value}</div>
    </div>
  );
}

/* ---------- CHARTS ---------- */
function Charts({ data }: { data: DashboardData }) {
  const chartData = data.episodes.map((ep: number, i: number) => ({
    episode: ep,
    avg_wait: data.avg_wait[i],
    throughput: data.throughput[i],
    queue_length: data.queue_length[i],
    mse: data.mse[i],
  }));

  return (
    <div className="charts">
      <div className="chart-card">
        <h3 className="chart-title">Efficiency by Episode</h3>
        <div className="chart-frame">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={chartData} margin={{ top: 8, right: 16, left: 8, bottom: 18 }}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="episode">
                <Label value="Episode" offset={-10} position="insideBottom" />
              </XAxis>
              <YAxis>
                <Label
                  value="Value"
                  angle={-90}
                  position="insideLeft"
                  style={{ textAnchor: "middle" }}
                />
              </YAxis>
              <Tooltip />
              <Legend verticalAlign="top" height={36} />
              <Line
                type="monotone"
                dataKey="avg_wait"
                name="Avg Wait"
                stroke="#e17f25"
                strokeWidth={3}
                dot={false}
              />
              <Line
                type="monotone"
                dataKey="throughput"
                name="Throughput"
                stroke="#2e7d32"
                strokeWidth={3}
                dot={false}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div className="chart-card">
        <h3 className="chart-title">Congestion by Episode</h3>
        <div className="chart-frame">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={chartData} margin={{ top: 8, right: 16, left: 8, bottom: 18 }}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="episode">
                <Label value="Episode" offset={-10} position="insideBottom" />
              </XAxis>
              <YAxis>
                <Label
                  value="Value"
                  angle={-90}
                  position="insideLeft"
                  style={{ textAnchor: "middle" }}
                />
              </YAxis>
              <Tooltip />
              <Legend verticalAlign="top" height={36} />
              <Line
                type="monotone"
                dataKey="queue_length"
                name="Queue Length"
                stroke="#1565c0"
                strokeWidth={3}
                dot={false}
              />
              <Line
                type="monotone"
                dataKey="mse"
                name="MSE"
                stroke="#8e24aa"
                strokeWidth={3}
                dot={false}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
}

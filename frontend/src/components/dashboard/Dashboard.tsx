import { Link } from "react-router-dom";
import "./Dashboard.css";

const corridorMetrics = [
  { label: "Average Wait", value: "42s", detail: "-18% from last hour" },
  { label: "Vehicles Routed", value: "12,480", detail: "Across 6 intersections" },
  { label: "Signal Efficiency", value: "94.1%", detail: "+2.4 pts this shift" },
];

const activeAlerts = [
  {
    title: "Downtown East",
    description: "Heavy queue detected near the northbound turn lane.",
    severity: "High",
  },
  {
    title: "Campus Loop",
    description: "Adaptive timing applied after pedestrian surge.",
    severity: "Moderate",
  },
  {
    title: "River Crossing",
    description: "Sensors reporting stable flow after retiming.",
    severity: "Low",
  },
];

function Dashboard() {
  return (
    <main className="dashboard-page">
      <header className="dashboard-hero">
        <div>
          <p className="dashboard-kicker">Live Operations</p>
          <h1>Traffic Reinforcement Learning Dashboard</h1>
          <p className="dashboard-subtitle">
            Overview of routing performance, active incidents, and signal health.
          </p>
        </div>
        <Link className="dashboard-home-link" to="/">
          Back Home
        </Link>
      </header>

      <section className="metrics-grid" aria-label="Traffic metrics">
        {corridorMetrics.map((metric) => (
          <article className="metric-card" key={metric.label}>
            <p>{metric.label}</p>
            <strong>{metric.value}</strong>
            <span>{metric.detail}</span>
          </article>
        ))}
      </section>

      <section className="dashboard-panels">
        <article className="panel panel-wide">
          <div className="panel-header">
            <div>
              <p className="panel-label">Network Status</p>
              <h2>Intersection Health</h2>
            </div>
            <span className="status-pill">All systems nominal</span>
          </div>
          <div className="health-grid">
            <div>
              <span>Autonomous Mode</span>
              <strong>5 / 6 corridors</strong>
            </div>
            <div>
              <span>Manual Override</span>
              <strong>1 corridor</strong>
            </div>
            <div>
              <span>Emergency Preemption</span>
              <strong>0 active events</strong>
            </div>
            <div>
              <span>Sensor Uptime</span>
              <strong>99.4%</strong>
            </div>
          </div>
        </article>

        <article className="panel">
          <div className="panel-header">
            <div>
              <p className="panel-label">Alerts</p>
              <h2>Current Events</h2>
            </div>
          </div>
          <div className="alert-list">
            {activeAlerts.map((alert) => (
              <article className="alert-item" key={alert.title}>
                <div>
                  <h3>{alert.title}</h3>
                  <p>{alert.description}</p>
                </div>
                <span className="severity-badge">{alert.severity}</span>
              </article>
            ))}
          </div>
        </article>
      </section>
    </main>
  );
}

export default Dashboard;

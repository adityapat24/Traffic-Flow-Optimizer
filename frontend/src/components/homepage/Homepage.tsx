import { Link } from "react-router-dom";
import "./Homepage.css";

function HomePage() {
  return (
    <main className="app-shell">
      <section className="hero-card">
        <p className="eyebrow">Traffic RL Control Center</p>
        <h1>Monitor traffic intelligence and jump into the dashboard.</h1>
        <p className="hero-copy">
          Route management is set up and ready for feature pages. Use the dashboard to surface key
          metrics, incidents, and system status.
        </p>
        <div className="hero-actions">
          <Link className="primary-link" to="/dashboard">
            Open Dashboard
          </Link>
        </div>
      </section>
    </main>
  );
}

export default HomePage;

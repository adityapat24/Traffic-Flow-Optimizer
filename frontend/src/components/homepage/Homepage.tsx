import { Link } from "react-router-dom";
import "./Homepage.css";

function HomePage() {
  return (
    <main className="app-shell">
      <section className="hero-card">
        <p className="eyebrow">Traffic RL - CS4100 Artificial Intelligence</p>

        <h1>Compare traffic signal control agents using simulation results.</h1>

        <p className="hero-copy">
          Evaluate Fixed, Actuated, DQN, and PPO agents using key metrics like average wait time,
          throughput, queue length, and model error. Results are precomputed to enable fast,
          interactive comparison during the demo.
        </p>

        <div className="hero-actions">
          <Link className="primary-link" to="/dashboard">
            Open Dashboard
          </Link>
          <a
            className="secondary-link"
            href="https://github.com/adityapat24/CS4100_Final_Project"
            target="_blank"
            rel="noreferrer"
          >
            View GitHub Repo
          </a>
        </div>

        <div className="created-by">
          <span>Created by:</span>
          <p>Aditya Pathak, Alex Kouyoumjian, Charles Heese, Holt Moriarty</p>
        </div>
      </section>
    </main>
  );
}

export default HomePage;

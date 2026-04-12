# CS4100_Final_Project

Final Project For CS4100

## Aditya, Alex, Charles, Holt

## Run the Demo Web App

### Local

- Go to /frontend and run `npm run dev`
- Go to `http://localhost:5173/` to view and interact with the app

### Website URL

- Go to `https://cs4100finalproject-git-main-akouyoumjians-projects.vercel.app/`

## Run the Live Sim (Dashboard "Live Sim" tab)

The Live Sim tab streams a real SUMO simulation to the dashboard and shows the agent's decisions in real time. It requires a local backend server.

**1. Install backend dependencies**
```bash
cd traffic-rl
pip install fastapi "uvicorn[standard]" websockets
```

**2. Start the backend**
```bash
cd traffic-rl
uvicorn api.server:app --port 8000
```

**3. Start the frontend** (separate terminal)
```bash
cd frontend
npm run dev
```

**4. Open the dashboard** at `http://localhost:5173/dashboard`, click the **Live Sim** tab, select an agent, and hit **Start Simulation**.

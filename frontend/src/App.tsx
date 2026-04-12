import { Navigate, Route, Routes } from "react-router-dom";
import Dashboard from "./components/dashboard/Dashboard";
import HomePage from "./components/homepage/Homepage";

function App() {
  return (
    <Routes>
      <Route path="/" element={<HomePage />} />
      <Route path="/dashboard" element={<Dashboard />} />
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  );
}

export default App;

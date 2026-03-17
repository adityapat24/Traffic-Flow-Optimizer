import os
import sys
import traci


def setup_sumo_tools():
    if "SUMO_HOME" in os.environ:
        tools = os.path.join(os.environ["SUMO_HOME"], "tools")
        if tools not in sys.path:
            sys.path.append(tools)


def main():
    setup_sumo_tools()
    sumocfg_path = "sumo/sim.sumocfg"

    try:
        traci.start(["sumo", "-c", sumocfg_path])

        tls_ids = traci.trafficlight.getIDList()
        if not tls_ids:
            raise RuntimeError("No traffic lights found in the simulation.")

        tl_id = tls_ids[0]
        lane_ids = list(dict.fromkeys(traci.trafficlight.getControlledLanes(tl_id)))

        if not lane_ids:
            raise RuntimeError(f"No controlled lanes found for traffic light '{tl_id}'.")

        print("=" * 70)
        print("Connected to SUMO via TraCI")
        print(f"Traffic light ID: {tl_id}")
        print(f"Controlled lanes: {lane_ids}")
        print("=" * 70)

        for step in range(100):
            traci.simulationStep()
            phase = traci.trafficlight.getPhase(tl_id)

            print(f"\nStep {step + 1}")
            print(f"Traffic light phase: {phase}")
            print("-" * 70)

            for lane_id in lane_ids:
                vehicle_count = traci.lane.getLastStepVehicleNumber(lane_id)
                waiting_time = traci.lane.getWaitingTime(lane_id)

                print(
                    f"Lane: {lane_id:<20} "
                    f"Vehicles: {vehicle_count:<3} "
                    f"Waiting Time: {waiting_time:.2f}"
                )

        print("\nFinished 100 simulation steps.")

    finally:
        traci.close()
        print("SUMO connection closed cleanly.")


if __name__ == "__main__":
    main()

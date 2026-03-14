import traci

sumo_cmd = ["sumo", "-c", "sumo/test/grid6.sumocfg"]

traci.start(sumo_cmd)

for i in range(10):
    traci.simulationStep()

print("Step 10 complete.")

traci.close()
print("SUMO connection closed.")

import random

SIMULATION_STEPS = 50 
MIN_PHASE_DURATION = 10 
MAX_PHASE_DURATION = 30
EMERGENCY_OVERRIDE_DURATION = 5

PHASES = ["NS_GREEN", "EW_GREEN"]

class TrafficAgent:
    def __init__(self):
        self.queues = {
            "N": 5,
            "S": 5,
            "E": 5,
            "W": 5
        }
        
        self.current_phase = PHASES[0]
        self.time_in_phase = 0
        self.current_duration = MIN_PHASE_DURATION
        
        self.emergency_present = False
        self.emergency_timer = 0
        self.is_header_printed = False
        
        print("TrafficAgent initialized. Intersection ready for adaptive control.")

    def _simulate_traffic_flow(self, active_phase):
        cleared_count = 0
        
        if active_phase == "NS_GREEN":
            cleared_count += random.randint(3, 7)
            self.queues['N'] = max(0, self.queues['N'] - cleared_count)
            self.queues['S'] = max(0, self.queues['S'] - cleared_count)
            self.queues['E'] += random.randint(0, 3)
            self.queues['W'] += random.randint(0, 3)
            
        elif active_phase == "EW_GREEN":
            cleared_count += random.randint(3, 7)
            self.queues['E'] = max(0, self.queues['E'] - cleared_count)
            self.queues['W'] = max(0, self.queues['W'] - cleared_count)
            self.queues['N'] += random.randint(0, 3)
            self.queues['S'] += random.randint(0, 3)
            
        for lane in self.queues:
            self.queues[lane] = min(self.queues[lane], 30)

    def _check_for_emergency(self):
        if self.emergency_present:
            self.emergency_timer -= 1
            if self.emergency_timer <= 0:
                self.emergency_present = False
        else:
            if random.random() < 0.05:
                self.emergency_present = True
                
    def _calculate_adaptive_duration(self, current_phase):
        if current_phase == "NS_GREEN":
            total_queue = self.queues['N'] + self.queues['S']
        else:
            total_queue = self.queues['E'] + self.queues['W']

        scale_factor = total_queue / 60.0
        
        duration = MIN_PHASE_DURATION + (MAX_PHASE_DURATION - MIN_PHASE_DURATION) * scale_factor
        
        return round(max(MIN_PHASE_DURATION, min(MAX_PHASE_DURATION, duration)))

    def step(self):
        
        self.time_in_phase += 1
        
        if self.emergency_present:
            target_phase = "NS_GREEN" if self.queues['N'] > self.queues['E'] else "EW_GREEN"
            
            if self.current_phase != target_phase:
                self.current_phase = target_phase
                self.current_duration = EMERGENCY_OVERRIDE_DURATION
                self.time_in_phase = 0
            
            elif self.time_in_phase >= self.current_duration:
                 self.emergency_present = False
                 self.emergency_timer = 0
                 
        elif self.time_in_phase >= self.current_duration:
            
            if self.current_phase == "NS_GREEN":
                self.current_phase = "EW_GREEN"
            else:
                self.current_phase = "NS_GREEN"
            
            self.time_in_phase = 0
            
            self.current_duration = self._calculate_adaptive_duration(self.current_phase)

        self._simulate_traffic_flow(self.current_phase)
        
        self._check_for_emergency()

    def display_state(self, step):
        
        if not self.is_header_printed:
            print("\n" + "="*80)
            print(f"{'STEP':<4} | {'PHASE':<8} | {'TIME/DUR':<8} | {'EMERGENCY':<9} | {'Q_N':<3} | {'Q_S':<3} | {'Q_E':<3} | {'Q_W':<3}")
            print("="*80)
            self.is_header_printed = True

        status = "YES" if self.emergency_present else "NO"
        print(f"{step:<4} | {self.current_phase:<8} | {self.time_in_phase:>2}/{self.current_duration:<5} | {status:<9} | {self.queues['N']:<3} | {self.queues['S']:<3} | {self.queues['E']:<3} | {self.queues['W']:<3}")


if __name__ == "__main__":
    agent = TrafficAgent()
    
    for t in range(1, SIMULATION_STEPS + 1):
        agent.step()
        agent.display_state(t)
        
        if t == SIMULATION_STEPS:
            break
        
    print("\n--- Traffic Signal Control Simulation Complete ---")
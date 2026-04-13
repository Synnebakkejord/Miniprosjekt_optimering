import simpy
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

plt.style.use('ggplot')

# ── Configuration ──────────────────────────────────────────────────────────────
INPUT_FILE        = 'Input miniprosjekt.xlsx'
NUM_REPLICATIONS  = 20
NUM_WEEKS         = 5
WARMUP_WEEKS      = 1
NUM_PATIENT_TYPES = 3   # 0 = red (most urgent), 1 = yellow, 2 = green

NUM_TRIAGE_BEDS = 10
NUM_NURSES      = 5
NUM_DOCTORS     = 5

TRIAGE_MEAN_MIN = 10    # mean triage duration (minutes)
DOCTOR_MEAN_MIN = 60    # mean diagnostics duration (minutes)
TRIAGE_RATE     = 1 / TRIAGE_MEAN_MIN
DOCTOR_RATE     = 1 / DOCTOR_MEAN_MIN

RANDOM_SEED         = 10
PATIENT_TYPE_NAMES  = ['Red (urgent)', 'Yellow', 'Green']
WARMUP_MINUTES      = WARMUP_WEEKS * 7 * 24 * 60

# ── Load and reshape arrival rates from Excel ──────────────────────────────────
# arrival_rates[patient_type, day_of_week, hour_of_day]
input_data     = pd.read_excel(INPUT_FILE, sheet_name='Sheet1')
arrival_rates  = input_data['Lambda'].values.reshape(NUM_PATIENT_TYPES, 7, 24)

# Maximum rate per type — used for the thinning (rejection-sampling) algorithm
max_rate_per_type = arrival_rates.reshape(NUM_PATIENT_TYPES, -1).max(axis=1)

# ── Results storage ────────────────────────────────────────────────────────────
# Waiting time to triage (post warm-up)
triage_waits      = [[] for _ in range(NUM_REPLICATIONS)]
triage_waits_type = [[[] for _ in range(NUM_PATIENT_TYPES)]
                     for _ in range(NUM_REPLICATIONS)]

# Waiting time for doctor (post warm-up)
doctor_waits      = [[] for _ in range(NUM_REPLICATIONS)]
doctor_waits_type = [[[] for _ in range(NUM_PATIENT_TYPES)]
                     for _ in range(NUM_REPLICATIONS)]

# Emergency bed occupancy — sampled at each bed-entry and bed-exit event
# (post warm-up). Shape: (replications, day_of_week, hour_of_day)
bed_occ_sum   = np.zeros((NUM_REPLICATIONS, 7, 24))
bed_occ_count = np.zeros((NUM_REPLICATIONS, 7, 24))

# ── Patient class ──────────────────────────────────────────────────────────────
class Patient:
    """A single patient visiting the emergency department."""
    def __init__(self, patient_id: int, patient_type: int):
        self.patient_id   = patient_id
        self.patient_type = patient_type   # 0=red, 1=yellow, 2=green

# ── Helper functions ───────────────────────────────────────────────────────────
def in_steady_state(env) -> bool:
    return env.now >= WARMUP_MINUTES

def record_bed_occupancy(env, run_idx: int, current_count: int):
    """Snapshot the current emergency bed count at this simulation time."""
    if not in_steady_state(env):
        return
    t    = env.now
    day  = int((t // (60 * 24)) % 7)
    hour = int((t // 60) % 24)
    bed_occ_sum[run_idx, day, hour]   += current_count
    bed_occ_count[run_idx, day, hour] += 1

# ── Patient process ────────────────────────────────────────────────────────────
def patient_process(env, patient: Patient, run_idx: int,
                    triage_beds, nurses, doctors):
    global emergency_beds_occupied

    # Stage 1 — Triage: needs a triage bed AND a nurse at the same time
    entered_triage_queue = env.now

    with triage_beds.request() as req_bed, nurses.request() as req_nurse:
        yield req_bed & req_nurse

        triage_wait = env.now - entered_triage_queue
        if in_steady_state(env):
            triage_waits[run_idx].append(triage_wait)
            triage_waits_type[run_idx][patient.patient_type].append(triage_wait)

        yield env.timeout(random.expovariate(TRIAGE_RATE))
    # Triage bed and nurse are released here

    # Stage 2 — Emergency bed (held until departure) + Doctor visit
    emergency_beds_occupied += 1
    record_bed_occupancy(env, run_idx, emergency_beds_occupied)

    entered_doctor_queue = env.now

    with doctors.request(priority=patient.patient_type) as req_doctor:
        yield req_doctor

        doctor_wait = env.now - entered_doctor_queue
        if in_steady_state(env):
            doctor_waits[run_idx].append(doctor_wait)
            doctor_waits_type[run_idx][patient.patient_type].append(doctor_wait)

        yield env.timeout(random.expovariate(DOCTOR_RATE))

    # Patient departs — release the emergency bed
    emergency_beds_occupied -= 1
    record_bed_occupancy(env, run_idx, emergency_beds_occupied)

# ── Arrival process — non-homogeneous Poisson via thinning ────────────────────
def arrival_process(env, num_weeks: int, run_idx: int,
                    triage_beds, nurses, doctors):
    """
    Generates arrivals using the thinning (Lewis-Shedler) algorithm.
    Each patient type is tracked independently; the earliest next arrival
    across all types is processed first.
    """
    sim_end       = 60 * 24 * 7 * num_weeks
    patient_count = 0

    # Next scheduled arrival time per patient type
    next_arrival_per_type = np.zeros(NUM_PATIENT_TYPES)

    while True:
        current_time = np.min(next_arrival_per_type)
        if current_time >= sim_end:
            break

        # Find the type with the earliest next arrival
        patient_type = int(np.argmin(next_arrival_per_type))

        # Dispatch this patient at the current simulation time
        patient = Patient(patient_count, patient_type)
        env.process(patient_process(env, patient, run_idx,
                                    triage_beds, nurses, doctors))
        patient_count += 1

        # Thinning: find the next accepted arrival for this patient type
        t        = current_time
        accepted = False
        while not accepted and t < sim_end:
            t   += random.expovariate(max_rate_per_type[patient_type])
            day  = int((t // (60 * 24)) % 7)
            hour = int((t // 60) % 24)
            rate = arrival_rates[patient_type, day, hour]
            if t < sim_end and random.random() <= rate / max_rate_per_type[patient_type]:
                accepted = True

        next_arrival_per_type[patient_type] = t if accepted else sim_end

        # Advance clock to the next soonest arrival across all types
        yield env.timeout(np.min(next_arrival_per_type) - current_time)

# ── Main simulation loop ───────────────────────────────────────────────────────
for run in range(NUM_REPLICATIONS):
    print(f'Replication {run + 1} / {NUM_REPLICATIONS}')
    emergency_beds_occupied = 0

    random.seed(RANDOM_SEED + run)
    env = simpy.Environment()

    triage_beds = simpy.Resource(env, capacity=NUM_TRIAGE_BEDS)
    nurses      = simpy.Resource(env, capacity=NUM_NURSES)
    doctors     = simpy.PriorityResource(env, capacity=NUM_DOCTORS)

    env.process(arrival_process(env, NUM_WEEKS, run, triage_beds, nurses, doctors))
    env.run()

print('Simulation complete.\n')

# ── Statistics helpers ─────────────────────────────────────────────────────────
def ci95_halfwidth(values: np.ndarray) -> float:
    """Half-width of a 95% confidence interval."""
    return 1.96 * np.std(values, ddof=1) / math.sqrt(len(values))

def print_wait_stats(label: str, wait_lists: list):
    """
    Print mean wait, 95th-percentile wait, and 95% CI for both,
    computed across replications (one value per replication).
    """
    run_means = np.array([np.mean(w) for w in wait_lists if w])
    run_p95   = np.array([np.percentile(w, 95) for w in wait_lists if w])

    mean_val = np.mean(run_means);  hw_mean = ci95_halfwidth(run_means)
    p95_val  = np.mean(run_p95);    hw_p95  = ci95_halfwidth(run_p95)

    print(f'  {label}')
    print(f'    Mean wait:       {mean_val:6.2f} min   '
          f'95% CI [{mean_val - hw_mean:.2f}, {mean_val + hw_mean:.2f}]')
    print(f'    95th percentile: {p95_val:6.2f} min   '
          f'95% CI [{p95_val - hw_p95:.2f}, {p95_val + hw_p95:.2f}]')

# ── Plot: Emergency bed occupancy ──────────────────────────────────────────────
avg_beds_per_run = np.zeros((NUM_REPLICATIONS, 7 * 24))
for run in range(NUM_REPLICATIONS):
    for d in range(7):
        for h in range(24):
            n = bed_occ_count[run, d, h]
            if n > 0:
                avg_beds_per_run[run, d * 24 + h] = bed_occ_sum[run, d, h] / n

mean_beds = np.mean(avg_beds_per_run, axis=0)
std_beds  = np.std(avg_beds_per_run, axis=0, ddof=1)
ci_upper  = mean_beds + 1.96 * std_beds / math.sqrt(NUM_REPLICATIONS)
ci_lower  = mean_beds - 1.96 * std_beds / math.sqrt(NUM_REPLICATIONS)

hours      = np.arange(168)
x_ticks    = np.arange(0, 168, 24)
day_labels = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

plt.figure(figsize=(12, 5))
plt.plot(hours, mean_beds, 'k', label='Mean')
plt.fill_between(hours, ci_lower, ci_upper, alpha=0.3, color='gray', label='95% CI')
plt.xlabel('Hour of week')
plt.ylabel('Emergency beds occupied')
plt.title(f'Emergency bed occupancy — {NUM_REPLICATIONS} replications')
plt.xticks(x_ticks, day_labels)
plt.legend()
plt.tight_layout()
plt.show()

# ── Task 3c — Triage waiting time statistics ───────────────────────────────────
print('=== Task 3c — Triage waiting time (all patients) ===')
print_wait_stats('All patients', triage_waits)

# ── Task 3d — Doctor waiting time by patient type ──────────────────────────────
print('\n=== Task 3d — Doctor waiting time by patient type ===')
for ptype in range(NUM_PATIENT_TYPES):
    per_run = [doctor_waits_type[run][ptype] for run in range(NUM_REPLICATIONS)]
    print_wait_stats(PATIENT_TYPE_NAMES[ptype], per_run)

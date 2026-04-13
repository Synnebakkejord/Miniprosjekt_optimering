import simpy
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

plt.style.use('ggplot')

# Configuration
INPUT_FILE        = 'Input miniprosjekt.xlsx'
NUM_WEEKS         = 5
WARMUP_WEEKS      = 1
NUM_PATIENT_TYPES = 3   # 0 = red (most urgent), 1 = yellow, 2 = green

NUM_TRIAGE_BEDS = 10
NUM_NURSES      = 5
NUM_DOCTORS     = 5

TRIAGE_MEAN_MIN = 10    # mean triage duration (minutes)

# Task 3 doctor parameters
DOCTOR_MEAN_MIN_T3 = 60   # mean diagnostics duration (minutes)

# Task 4 doctor parameters
DOCTOR_MEAN_MIN_T4 = 30   # mean diagnostics duration (minutes)
BLOOD_TEST_PROB    = 0.40  # probability a patient needs a blood test
EVAL_DURATION_MIN  = 5    # deterministic evaluation duration after blood test results

RANDOM_SEED        = 10
PATIENT_TYPE_NAMES = ['Red (urgent)', 'Yellow', 'Green']
WARMUP_MINUTES     = WARMUP_WEEKS * 7 * 24 * 60



# Load and reshape arrival rates from Excel
# arrival_rates[patient_type, day_of_week, hour_of_day]
input_data    = pd.read_excel(INPUT_FILE, sheet_name='Sheet1')
arrival_rates = input_data['Lambda'].values.reshape(NUM_PATIENT_TYPES, 7, 24)

# Maximum rate per type — used for the thinning (rejection-sampling) algorithm
max_rate_per_type = arrival_rates.reshape(NUM_PATIENT_TYPES, -1).max(axis=1)



# Patient class
class Patient:
    """A single patient visiting the emergency department."""
    def __init__(self, patient_id: int, patient_type: int):
        self.patient_id   = patient_id
        self.patient_type = patient_type   # 0=red, 1=yellow, 2=green

# Patient process 
def patient_process(env, patient: Patient, run_idx: int, triage_beds, nurses, doctors,
                    results: dict, bed_counter: list, doctor_mean_min: float, blood_test_model: bool):

    doctor_rate = 1 / doctor_mean_min

    # Stage 1 — Triage: needs a triage bed AND a nurse simultaneously
    entered_triage_queue = env.now

    with triage_beds.request() as req_bed, nurses.request() as req_nurse:
        yield req_bed & req_nurse

        triage_wait = env.now - entered_triage_queue
        if env.now >= WARMUP_MINUTES:
            results['triage_waits'][run_idx].append(triage_wait)
            results['triage_waits_type'][run_idx][patient.patient_type].append(triage_wait)

        yield env.timeout(random.expovariate(1 / TRIAGE_MEAN_MIN))
    # Triage bed and nurse are released here

    # Stage 2 — Emergency bed (held until departure) + Doctor visit
    bed_counter[0] += 1
    _record_bed_occupancy(env, run_idx, bed_counter[0], results)

    entered_doctor_queue = env.now

    with doctors.request(priority=patient.patient_type) as req_doctor:
        yield req_doctor

        doctor_wait = env.now - entered_doctor_queue
        if env.now >= WARMUP_MINUTES:
            results['doctor_waits'][run_idx].append(doctor_wait)
            results['doctor_waits_type'][run_idx][patient.patient_type].append(doctor_wait)

        yield env.timeout(random.expovariate(doctor_rate))

    # Stage 3 (task 4 only) — Blood test for 40% of patients
    if blood_test_model and random.random() < BLOOD_TEST_PROB:
        # Wait until the next whole-hour batch send
        t          = env.now
        next_batch = math.ceil(t / 60) * 60
        if next_batch == t:        # exactly on the hour — next batch is 1h away
            next_batch += 60
        yield env.timeout(next_batch - t)

        # Results arrive 1 hour after the batch is sent
        yield env.timeout(60)

        # Doctor evaluates the result (priority-based, deterministic 5 min)
        with doctors.request(priority=patient.patient_type) as req_eval:
            yield req_eval
            yield env.timeout(EVAL_DURATION_MIN)

    # Patient departs — release the emergency bed
    bed_counter[0] -= 1
    _record_bed_occupancy(env, run_idx, bed_counter[0], results)

def _record_bed_occupancy(env, run_idx: int, current_count: int, results: dict):
    if env.now < WARMUP_MINUTES:
        return
    t    = env.now
    day  = int((t // (60 * 24)) % 7)
    hour = int((t // 60) % 24)
    results['bed_occ_sum'][run_idx, day, hour]   += current_count
    results['bed_occ_count'][run_idx, day, hour] += 1

# Arrival process — non-homogeneous Poisson via thinning
def arrival_process(env, num_weeks: int, run_idx: int, triage_beds, nurses, doctors,
                    results: dict, bed_counter: list, doctor_mean_min: float, blood_test_model: bool):

    sim_end       = 60 * 24 * 7 * num_weeks
    patient_count = 0
    next_arrival_per_type = np.zeros(NUM_PATIENT_TYPES)

    while True:
        current_time = float(np.min(next_arrival_per_type))
        if current_time >= sim_end:
            break

        patient_type = int(np.argmin(next_arrival_per_type))

        patient = Patient(patient_count, patient_type)
        env.process(patient_process(env, patient, run_idx,
                                    triage_beds, nurses, doctors,
                                    results, bed_counter,
                                    doctor_mean_min, blood_test_model))
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

        yield env.timeout(float(np.min(next_arrival_per_type)) - current_time)

# Simulation runner
def run_simulation(num_replications: int, num_doctors: int = NUM_DOCTORS, 
                   doctor_mean_min: float = DOCTOR_MEAN_MIN_T3, blood_test_model: bool = False) -> dict:

    results = {
        'num_replications': num_replications,
        'num_doctors':      num_doctors,
        'triage_waits':      [[] for _ in range(num_replications)],
        'triage_waits_type': [[[] for _ in range(NUM_PATIENT_TYPES)]
                              for _ in range(num_replications)],
        'doctor_waits':      [[] for _ in range(num_replications)],
        'doctor_waits_type': [[[] for _ in range(NUM_PATIENT_TYPES)]
                              for _ in range(num_replications)],
        'bed_occ_sum':   np.zeros((num_replications, 7, 24)),
        'bed_occ_count': np.zeros((num_replications, 7, 24)),
    }

    for run in range(num_replications):
        print(f'  Replication {run + 1} / {num_replications}')
        bed_counter = [0]

        random.seed(RANDOM_SEED + run)
        env = simpy.Environment()

        triage_beds = simpy.Resource(env, capacity=NUM_TRIAGE_BEDS)
        nurses      = simpy.Resource(env, capacity=NUM_NURSES)
        doctors     = simpy.PriorityResource(env, capacity=num_doctors)

        env.process(arrival_process(env, NUM_WEEKS, run,
                                    triage_beds, nurses, doctors,
                                    results, bed_counter,
                                    doctor_mean_min, blood_test_model))
        env.run()

    return results

# Statistics helpers 
def ci95_halfwidth(values: np.ndarray) -> float:
    """Half-width of a 95% confidence interval (normal approximation)."""
    return float(1.96 * np.std(values, ddof=1) / math.sqrt(len(values)))

def bed_occupancy_curve(results: dict):
    """Return (hours, mean, ci_lower, ci_upper) for the emergency bed occupancy."""
    n   = results['num_replications']
    avg = np.zeros((n, 7 * 24))
    for run in range(n):
        for d in range(7):
            for h in range(24):
                count = results['bed_occ_count'][run, d, h]
                if count > 0:
                    avg[run, d * 24 + h] = results['bed_occ_sum'][run, d, h] / count

    mean      = np.mean(avg, axis=0)
    halfwidth = 1.96 * np.std(avg, axis=0, ddof=1) / math.sqrt(n)
    return np.arange(168), mean, mean - halfwidth, mean + halfwidth

def print_wait_stats(label: str, wait_lists: list):
    """Print mean, 95th-percentile wait, and 95% CI for both across replications."""
    run_means = np.array([np.mean(w) for w in wait_lists if w])
    run_p95   = np.array([np.percentile(w, 95) for w in wait_lists if w])

    m  = float(np.mean(run_means));  hw_m = ci95_halfwidth(run_means)
    p  = float(np.mean(run_p95));    hw_p = ci95_halfwidth(run_p95)

    print(f'  {label}')
    print(f'    Mean wait:       {m:6.2f} min   95% CI [{m - hw_m:.2f}, {m + hw_m:.2f}]')
    print(f'    95th percentile: {p:6.2f} min   95% CI [{p - hw_p:.2f}, {p + hw_p:.2f}]')

def plot_bed_occupancy(ax, results: dict, title: str):
    hours, mean, lo, hi = bed_occupancy_curve(results)
    x_ticks    = np.arange(0, 168, 24)
    day_labels = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    ax.plot(hours, mean, 'k', label='Mean')
    ax.fill_between(hours, lo, hi, alpha=0.3, color='steelblue', label='95% CI')
    ax.set_ylabel('Emergency beds occupied')
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(day_labels)
    ax.set_title(title)
    ax.legend()

# Task 3 — original model (60 min diagnostics, no blood tests)
print('── Task 3: Running 20 replications ──')
results_t3_20 = run_simulation(20, doctor_mean_min=DOCTOR_MEAN_MIN_T3)

print('\n── Task 3: Running 100 replications ──')
results_t3_100 = run_simulation(100, doctor_mean_min=DOCTOR_MEAN_MIN_T3)

print()

# Task 3b — bed occupancy: 20 vs 100 replications
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
plot_bed_occupancy(ax1, results_t3_20,  'Task 3b — 20 replications')
plot_bed_occupancy(ax2, results_t3_100, 'Task 3b — 100 replications')
ax2.set_xlabel('Hour of week')
fig.suptitle('Task 3b — Emergency bed occupancy (original model)', fontsize=13)
plt.tight_layout()
plt.show()

# Task 3c — triage waiting time (100 replications)
print('=== Task 3c — Triage waiting time (all patients, 100 replications) ===')
print_wait_stats('All patients', results_t3_100['triage_waits'])

# Task 3d — doctor waiting time by type (100 replications)
print('\n=== Task 3d — Doctor waiting time by patient type (100 replications) ===')
for ptype in range(NUM_PATIENT_TYPES):
    per_run = [results_t3_100['doctor_waits_type'][run][ptype] for run in range(100)]
    print_wait_stats(PATIENT_TYPE_NAMES[ptype], per_run)

# Task 4 — blood test extension (30 min diagnostics, 40% blood tests)

NUM_DOCTORS_T4 = 5   # increase iteratively if system is unstable

print(f'\n── Task 4: Running 100 replications ({NUM_DOCTORS_T4} doctors) ──')
results_t4 = run_simulation(100,
                             num_doctors=NUM_DOCTORS_T4,
                             doctor_mean_min=DOCTOR_MEAN_MIN_T4,
                             blood_test_model=True)

# Task 4b — updated bed occupancy (100 replications)
fig, ax = plt.subplots(figsize=(13, 4))
plot_bed_occupancy(ax, results_t4,
                   f'Task 4b — Emergency bed occupancy (blood test model, '
                   f'{NUM_DOCTORS_T4} doctors, 100 replications)')
ax.set_xlabel('Hour of week')
plt.tight_layout()
plt.show()

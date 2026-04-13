import simpy
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

plt.style.use('ggplot')

# Leser input-data fra Excel
input_fil = pd.read_excel('Input miniprosjekt.xlsx', sheet_name='Sheet1')

# Parametere
Runs = 100
Weeks = 5
Warm_up = 1
Num_of_pat = 3

# Ressurskapasiteter
Num_sykepleiere = 5
Num_leger = 5
Num_triagesenger = 10

# Betjeningstider
lambda_triage = 1/10
lambda_lege = 1/60

# Leser lambda fra Excel
Lambda = input_fil['Lambda'].values

# Bygger lambda-tabell
Lambda_table = np.zeros((Num_of_pat, 7, 24))
teller = 0
for p in range(Num_of_pat):
    for d in range(7):
        for h in range(24):
            Lambda_table[p, d, h] = Lambda[teller]
            teller += 1

# Beregner Lambda_max automatisk
Lambda_max = np.zeros(Num_of_pat)
for p in range(Num_of_pat):
    Lambda_max[p] = np.max(Lambda_table[p])

# Datainnsamling
WT_triage = [[] for _ in range(Runs)]
WT_triage_type = [[[] for _ in range(3)] for _ in range(Runs)]
WT_lege = [[] for _ in range(Runs)]
WT_lege_type = [[[] for _ in range(3)] for _ in range(Runs)]
Senger_out = np.zeros((Runs, 7, 24))
Senger_count = np.zeros((Runs, 7, 24))

# Pasientklasse
class Patient:
    def __init__(self, env1, number, hastegrad, behandlingstid):
        self.env1 = env1
        self.number = number
        self.hastegrad = hastegrad
        self.behandlingstid = behandlingstid

# Ankomstfunksjon
def arrival(env1, weeks, lambda_time, num_of_pat):
    pat_count = 0
    next_arrivals = np.zeros(num_of_pat)
    next_arrival = 0

    while next_arrival < 60 * 24 * 7 * weeks:
        time = env1.now
        next_arrival = np.min(next_arrivals)

        for jj in range(len(next_arrivals)):
            if next_arrivals[jj] == next_arrival:
                break

        dur_behandling_pat = random.expovariate(lambda_lege)

        pat = Patient(env1, 'Patient %01d' % pat_count, jj, dur_behandling_pat)
        send_to_ed = ed(env1, patient=pat, run_nr=k)
        env1.process(send_to_ed)

        accept = 0
        time_count_test = next_arrival
        while accept == 0 and time_count_test < 60 * 24 * 7 * weeks:
            waiting_time = random.expovariate(Lambda_max[jj])
            time_count_test += waiting_time
            day = math.floor((time_count_test / (24 * 60)) % 7)
            hour = math.floor((time_count_test / 60) % 24)
            u = random.uniform(0, 1)
            if time_count_test < 60 * 24 * 7 * weeks and u <= lambda_time[jj][day][hour] / Lambda_max[jj]:
                accept = 1
                pat_count += 1

        next_arrivals[jj] = time_count_test
        next_pat_arrival = np.min(next_arrivals)
        yield env1.timeout(next_pat_arrival - next_arrival)

# Pasientprosess
def ed(env1, patient, run_nr):
    global pasienter_i_systemet

    arrive = env1.now
    triage_arrive = arrive
    pasienter_i_systemet += 1

    # Logg sengebelegg (kun etter oppvarming)
    week = int(math.floor((arrive / (24 * 60 * 7))))
    if week >= Warm_up:
        day = int(math.floor((arrive / (24 * 60)) % 7))
        hour = int(math.floor((arrive / 60) % 24))
        Senger_out[run_nr, day, hour] += pasienter_i_systemet
        Senger_count[run_nr, day, hour] += 1

    with resources[0].request() as req_seng, resources[1].request() as req_syk:
        yield req_seng & req_syk

        triage_start = env1.now
        ventetid_triage = triage_start - triage_arrive
        WT_triage[run_nr].append(ventetid_triage)
        WT_triage_type[run_nr][patient.hastegrad].append(ventetid_triage)

        yield env1.timeout(random.expovariate(lambda_triage))

    lege_arrive = env1.now
    with resources[2].request(priority=patient.hastegrad) as req_lege:
        yield req_lege

        lege_start = env1.now
        ventetid_lege = lege_start - lege_arrive
        WT_lege[run_nr].append(ventetid_lege)
        WT_lege_type[run_nr][patient.hastegrad].append(ventetid_lege)

        yield env1.timeout(random.expovariate(lambda_lege))

    pasienter_i_systemet -= 1

# Hovedløkke
Random_seed = 10

for k in range(Runs):
    print('Start simulating replication number: ', k)
    pasienter_i_systemet = 0

    random.seed(Random_seed + k)
    env = simpy.Environment()

    Triageseng = simpy.Resource(env, capacity=Num_triagesenger)
    Sykepleier = simpy.Resource(env, capacity=Num_sykepleiere)
    Lege = simpy.PriorityResource(env, capacity=Num_leger)

    resources = [Triageseng, Sykepleier, Lege]

    env.process(arrival(env, Weeks, Lambda_table, Num_of_pat))
    env.run()

print('Simulering ferdig!')

# Beregn gjennomsnittlig sengebelegg per time
Avg_senger = np.zeros((Runs, 7 * 24))
for run in range(Runs):
    for day in range(7):
        for hour in range(24):
            if Senger_count[run, day, hour] > 0:
                Avg_senger[run, day * 24 + hour] = Senger_out[run, day, hour] / Senger_count[run, day, hour]

# Beregn gjennomsnitt og 95% KI
mean_senger = np.mean(Avg_senger, axis=0)
std_senger = np.std(Avg_senger, axis=0)
KI_upper = mean_senger + 1.96 * std_senger / math.sqrt(Runs)
KI_lower = mean_senger - 1.96 * std_senger / math.sqrt(Runs)

# Plot
x_ticks = np.arange(0, 168, 24)
x_labels = ['Man', 'Tir', 'Ons', 'Tor', 'Fre', 'Lør', 'Søn']
plt.figure()
plt.plot(mean_senger, 'k', label='Gjennomsnitt')
plt.plot(KI_upper, 'k--', label='95% KI øvre')
plt.plot(KI_lower, 'k--', label='95% KI nedre')
plt.xlabel('Tid i uka')
plt.ylabel('Antall senger opptatt')
plt.title(f'Sengebelegg ({Runs} kjøringer)')
plt.xticks(x_ticks, x_labels)
plt.legend()
plt.show()
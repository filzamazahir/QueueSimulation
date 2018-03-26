# ECE610 Project - Queue Simulation
# Filza Mazahir - 20295951 - fmazahir@uwaterloo.ca


# Import libraries
import random
import numpy as np
from math import log, inf
from collections import deque

import matplotlib as mpl 
mpl.use('TkAgg')
import matplotlib.pyplot as plt

import time

# Open an output text file to output values for all figures
f = open('output.txt', 'w')



###########   Random Value Generator   ###########

# Test Python's built-in uniform random values generator
def test_uniform_random_value_generator():
	# Generate 1000 uniform random values
	uniform_rv_list = []
	for x in range(1000):
		uniform_random_value = random.uniform(0, 1) # Python's built-in uniform random generator
		uniform_rv_list.append(uniform_random_value)

	# Check with mean and variance - convert to numpy array and use np.mean and np.var
	uniform_rv_arr = np.array(uniform_rv_list)
	mean = np.mean(uniform_rv_arr)
	var = np.var(uniform_rv_arr)

	print('Uniform Random Value Generator with 1000 values:')
	print('Mean (uniform) - expected 0.5, calculated: {0:.3f}'.format(mean)) 
	print('Variance (uniform) - expected 0.0833 , calculated: {0:.3f}'.format(var))
	

# Generate Exponential Random Value - QUESTION 1
def generate_exponential_random_value(mu):
	uniform_rv = random.uniform(0, 1) # Python's built-in uniform random generator
	exponential_random_value = -(log(uniform_rv))/mu  # From equation shown in report
	return exponential_random_value
	# Python also has a random.expovariate(mu) function to generate exponential random values
	# But this function was written from the equation as per Question 1


# Generate 1000 Exponential Random Values to test - QUESTION 1
def generate_1000_exponential_random_values():
	mu = 0.1    # 1/mu = 10seconds -> mu = 0.1

	# Generate 1000 exponential random values
	exponential_rv_list = []
	for x in range(1000):
		exponential_rv = generate_exponential_random_value(mu)
		exponential_rv_list.append(exponential_rv)

	# Check with mean and variance - convert to numpy array and use np.mean and np.var
	exponential_rv_arr = np.array(exponential_rv_list)
	mean = np.mean(exponential_rv_arr)
	var = np.var(exponential_rv_arr)

	print('Exponential Random Value Generator with 1000 values:')
	print('Mean (exponential) - expected 10, calculated: {0:.1f}'.format(mean)) 
	f.write('Mean (exponential) - expected 10, calculated: {0:.1f}\n'.format(mean)) 
	print('Variance (exponential) - expected 100, calculated: {0:.1f}'.format(var))
	f.write('Variance (exponential) - expected 100, calculated: {0:.1f}\n'.format(var))
	return



###########   Simulation   ###########

class Simulation:
	def __init__(self, arrival_process, service_process, n, K, T, L, C, rho, L1, L2, prob):

		#Input variables
		self.arrival_process = arrival_process  # Arrival process ('M' or 'D')
		self.service_process = service_process  # Service process ('M', 'D' or 'G')
		self.n = n # Number of servers in the queue
		self.K = K  # Size of buffer (units-> no. of packets)

		self.T = T  # Total time to run the simulation
		self.L = L # Average packet length in bits
		self.C = C # Transmission rate in bits/second
		self.rho = rho # Utilization factor of the queue

		self.L1 = L1  # For General service process, L1 is the packet length with probability of prob
		self.L2 = L2 # For General service process, L2 is the packet length with probability of (1-prob)
		self.prob = prob  # For General service process, Probability of the packet to have L1 length

		self.lambd = (self.n * self.rho * self.C )/ self.L  # Arrival rate (avg no. of packets per sec)
		self.alpha = self.lambd  # Observer rate - same as arrival rate so they are in the same order  

		# State of the system 
		self.Na = 0 # Number of packet arrival events
		self.Nd = 0 # Number of packet departure events
		self.No = 0 # Number of observervation events
		self.Nt = 0 # Total number of packets in the system

		# Helper variables to keep track of things and determine the output variables
		self.Nt_observer = [] # No. of packets in system as seen by each observer event
		self.Nt_arrival = [] # No. of packets in system as seen by each arrival event
		self.Tsojourn = []  # Total sojourn time for each packet
		self.idle_counter = 0  # Number of times an observation event saw the system as idle
		self.total_packets_generated = 0 # Total number of packets generated in the system
		self.number_packets_dropped = 0  # Number of packets dropped by the queue
		
		print('Running simulation -> {0}/{1}/{2}/{3} (Rho = {4}, Lambda = {5})....'.format(self.arrival_process, self.service_process, self.n, self.K, self.rho, self.lambd), end='', flush=True)
		
		# Create Event Scheduler 
		self.create_event_scheduler()  


	# Function to generate arrival time based on the type of arrival process
	def generate_arrival_time(self):
		# Poisson Distribution for arrival process
		if self.arrival_process == 'M':  
			arrival_time = generate_exponential_random_value(self.lambd)

		# Deterministic Distribution for arrival process (constant)
		elif self.arrival_process == 'D':  
			arrival_time = (1.0/self.lambd)

		return arrival_time


	# Function to generate packet length based on type of service process
	def generate_packet_length(self):

		# Poisson Distribution for service process
		if self.service_process == 'M':
			packet_length = generate_exponential_random_value(1.0/self.L)

		# General Distribution with bipolar length for service process
		elif self.service_process == 'G':
			# Generate uniform random number between 0 and 1
			uniform_random_value = random.uniform(0, 1) 

			# Get packet length based on the number generated
			if uniform_random_value <= self.prob:
				packet_length = self.L1
			else:
				packet_length = self.L2

		# Deterministic Distribution for service process (constant length)
		elif self.service_process == 'D':
			packet_length = self.L

		return packet_length


	# Function to create Event Scheduler
	def create_event_scheduler(self):
		
		self.observation_times_list = []  # List of all observation times generated
		arrival_times_list = [] # List of all arrival times generated for each packet
		departure_times_list= []  # List of all depature times for each packet
		server_available_time = [0 for i in range(self.n)] # List of available time for all n servers

		# Generate set of random observation observation times with parameter alpha
		observation_time = 0  # Observation time initialization, starts at t = 0
		while (observation_time < T):
			observation_time += generate_exponential_random_value(self.alpha)
			self.observation_times_list.append(observation_time)


		# Generate packets, get packet length and its arrival and depature times
		arrival_time = 0   # Arrival time initialization, starts at t = 0
		while arrival_time < T:

			# Generate new packet - get arrival time and packet length
			arrival_time += self.generate_arrival_time()  # Generate arrival time
			packet_length = self.generate_packet_length() # Generate packet length
			self.total_packets_generated += 1 # Increment number of packets generated (was inititalized to 0)

			# Index of server that's available first (for 1 server queue, there is only 1 value here which is the minimum)
			i = server_available_time.index(min(server_available_time)) 

			# Functionality added for M/D/1/K queue - check if buffer is full
			# If not full - get departure time and sojourn time of this packet, and add to list
			# Length of depature_times_list will always be less than K when K=inf
			if len(departure_times_list) < self.K or departure_times_list[-self.K] < arrival_time:
				
				# Calculate the packet's departure time 
				# If server is free when the packet arrived (no wait)
				if server_available_time[i] <= arrival_time:
					departure_time = arrival_time + (packet_length/self.C)	

				# If server is not free when the packet arrived (packet has to wait in the queue)
				else:
					departure_time = server_available_time[i] + (packet_length/self.C)

				# Compute the soujorn time of the packet and save it
				sojourn_time = departure_time - arrival_time
				self.Tsojourn.append(sojourn_time)

				# Server will be available when this packet departs
				server_available_time[i] = departure_time 

				# Add the arrival event and departure event to Event Scheduler
				arrival_times_list.append(arrival_time)
				departure_times_list.append(departure_time)

			# If buffer is full - drop packet (don't add the arrival and depature time to list)
			else:
				self.number_packets_dropped += 1  # Increment number of dropped packets
			
				
		# Combine the list of all observation times, arrival times, and departure times in one list
		events_list = []  # List of all events combined

		# Add all observation events
		for event_time in self.observation_times_list:
			events_list.append(('O', event_time))

		# Add all arrival events
		for event_time in arrival_times_list:
			events_list.append(('A', event_time))

		# Add all depature events
		for event_time in departure_times_list:
			events_list.append(('D', event_time))
		
		# Sort events list based on time after its completed
		events_list.sort(key=lambda event: event[1])

		# Create Event Scheduler as a double ended queue from the events_list
		# Event Scheduler created after events_list is already sorted with time so its more efficient
		self.Event_Scheduler = deque(events_list)

		return


	# Function to run the simulation - goes through each event in the Event Scheduler 
	def run_simulation(self):

		# Loop through the Event Scheduler (continue looping as long as it has an element)
		# Update system metrics based on type of event
		while self.Event_Scheduler:
			# Dequeue the event from the Event Scheduler
			event_type, event_time = self.Event_Scheduler.popleft() 

			# Observation Event
			if event_type == 'O':
				self.No += 1  # Increment number of observation events
				self.Nt_observer.append(self.Nt)  # Record system metric

				# If system is idle, then increment the idle_counter
				if self.Nt == 0:
					self.idle_counter += 1

			# Arrival Event
			elif event_type == 'A':
				self.Nt_arrival.append(self.Nt) # Record system metric (excluding this packet)
				self.Na += 1  # Increment number of arrival events
				self.Nt = self.Na - self.Nd  # Update current number of packets in the system
					
			# Departure Event
			elif event_type == 'D':
				self.Nd += 1  # Increment number of departure events
				self.Nt = self.Na - self.Nd  # Update current number of packets in the system

		# Calculate output variables after the simulation is run and complete
		self.EN = sum(self.Nt_observer)/len(self.Nt_observer) # avg. no. of packets seen by obsever
		self.EaN = sum(self.Nt_arrival)/len(self.Nt_arrival) # aavg. no. of packets seen by arrival packet
		self.ET = sum(self.Tsojourn)/len(self.Tsojourn) # average of the sojourn time of all packets
		self.Pidle = self.idle_counter/self.No  # Proportion of time observer saw system idle 
		self.Ploss = self.number_packets_dropped/self.total_packets_generated  # Packet lost probability
	
		print('done')
		return


########################################################################################
#### Main function - This is where all functions are called and simulations are run ####

t0=time.time()



###########   QUESTION 1 - Exponential Random Value Generator   ###########
print('\nQUESTION 1')
f.write('\nQUESTION 1 - Exponential Random Value Generator with 1000 values\n')
test_uniform_random_value_generator()  #tests Python's built in uniform random generator
generate_1000_exponential_random_values() # uses-> generate_exponential_random_value(mu)




############   QUESTION 2 - Build simulator for M/M/1, D/M/1, and M/G/1   ###########
print('\nQUESTION 2 - Build simulator for M/M/1, D/M/1 and M/G/1')
# Simulator built above in class Simulation (lines 75-273). Testing for T (total simulation time) done here

# Initialize all variables to run the simulations as given in Question 3
T = 10000   # Total time for the simulation
L = 20000   # Average length of packets
C = 2000000 # Transmission rate of output link in bits/sec
rho = 0.5   # Utilization factor of queue
L1 = 16000  # For M/G/1 
L2 = 21000  # For M/G/1
prob = 0.2  # For M/G/1


# Check if the value of T gives stable result
print('Checking if the T being used is generating a stable system')
print('T ->', T)

# Run the simulation for M/M/1, D/M/1, and M/G/1 with rho = 0.5 and T = 10,000
MM1 = Simulation('M', 'M', 1, inf, T, L, C, rho, L1, L2, prob) # M/M/1
MM1.run_simulation()

DM1 = Simulation('D', 'M', 1, inf, T, L, C, rho, L1, L2, prob) # D/M/1
DM1.run_simulation()

MG1 = Simulation('M', 'G', 1, inf, T, L, C, rho, L1, L2, prob) # M/G/1
MG1.run_simulation()

# Run the simulation for M/M/1, D/M/1, and M/G/1 with rho = 0.5 and T = 20,000
T_double = 2*T
print('2T ->', T_double)
MM1_2T = Simulation('M', 'M', 1, inf, T_double, L, C, rho, L1, L2, prob) # M/M/1
MM1_2T.run_simulation()

DM1_2T = Simulation('D', 'M', 1, inf, T_double, L, C, rho, L1, L2, prob) # D/M/1
DM1_2T.run_simulation()

MG1_2T = Simulation('M', 'G', 1, inf, T_double, L, C, rho, L1, L2, prob) # M/G/1
MG1_2T.run_simulation()

# Compare the E[N] values and compute a ratio of the difference in values
MM1_ratio = (MM1.EN - MM1_2T.EN)/MM1.EN * 100
DM1_ratio = (DM1.EN - DM1_2T.EN)/DM1.EN * 100
MG1_ratio = (MG1.EN - MG1_2T.EN)/MG1.EN * 100

# Output the ratio computed 
print('Difference in values for M/M/1: {0:.1f}%'.format(MM1_ratio))
print('Difference in values for D/M/1: {0:.1f}%'.format(DM1_ratio))
print('Difference in values for M/G/1: {0:.1f}%'.format(MG1_ratio))

f.write('\nQUESTION 2 - Selecting T for M/M/1, D/M/1 and M/G/1 queue simulations\n')
f.write('Difference in values for M/M/1: {0:.1f}%\n'.format(MM1_ratio))
f.write('Difference in values for D/M/1: {0:.1f}%\n'.format(DM1_ratio))
f.write('Difference in values for M/G/1: {0:.1f}%\n'.format(MG1_ratio))

# Check if the ratio is less than 5% or not
if (MM1_ratio < 5 and DM1_ratio < 5 and MG1_ratio < 5):
	print('System is stable with T = {0} (values within 5%)'.format(T))
	f.write('System is stable with T = {0} (values within 5%)\n'.format(T))
else:
	print('System NOT stable with T = {0} (values differ more than 5%)'.format(T))
	f.write('System NOT stable with T = {0} (values differ more than 5%)\n'.format(T))




############   QUESTION 3 - Run simulation for the 3 queues for rho from 0.35 to 0.95 and plot graphs
print('\nQUESTION 3 - Run simulation for the 3 queues for rho from 0.35 to 0.95')
f.write('\nQUESTION 3 - E[N] and Pidle for 3 queues - 0.35 <= Rho <= 0.95 (Step size 0.05)\n')

# Empty lists to store values at each rho, to be used for plotting figures later
EN_MM1 = []    # E[N] values for M/M/1 - Question 3
Pidle_MM1 = [] # Pidle values for M/M/1 - Question 3
EaN_MM1 = []   # Ea[N] values for M/M/1 - Question 5
ET_MM1 = []    # E[T] values for M/M/1 - Question 6

EN_DM1 = []    # E[N] values for D/M/1 - Question 3
Pidle_DM1 = [] # Pidle values for D/M/1 - Question 3
EaN_DM1 = []   # Ea[N] values for D/M/1 - Question 5
ET_DM1 = []    # E[T] values for D/M/1 - Question 6

EN_MG1 = []    # E[N] values for M/G/1 - Question 3
Pidle_MG1 = [] # Pidle values for M/G/1 - Question 3
ET_MG1 = []    # E[T] values for M/G/1 - Question 6

rho_array = []  # List to store values of rho - to be used for x-axis when plotting graphs

# Loop for 0.35 <= Rho <= 0.95 (Step size 0.05)
for a in np.arange(0.35, 1.0, 0.05):
	rho = round(a, 2)
	rho_array.append(rho)  # Store value of rho - to be used for plotting graphs

	# M/M/1 Simulations for 0.35 <= Rho <= 0.95
	MM1 = Simulation('M', 'M', 1, inf, T, L, C, rho, L1, L2, prob)
	MM1.run_simulation()
	EN_MM1.append(MM1.EN)  # Question 3 - E[N] vs Rho figures
	Pidle_MM1.append(MM1.Pidle)  # Question 3 - Pidle vs Rho figures
	EaN_MM1.append(MM1.EaN)  # Question 5 - comparison of E[N] and EaN figures
	ET_MM1.append(MM1.ET)  # Question 6 - E[T] vs Rho figures

	# D/M/1 Simulations for 0.35 <= Rho <= 0.95
	DM1 = Simulation('D', 'M', 1, inf, T, L, C, rho, L1, L2, prob)
	DM1.run_simulation()
	EN_DM1.append(DM1.EN)  # Question 3 - E[N] vs Rho figures
	Pidle_DM1.append(DM1.Pidle)  # Question 3 - Pidle vs Rho figures
	EaN_DM1.append(DM1.EaN)  # Question 5 - comparison of E[N] and EaN figures
	ET_DM1.append(DM1.ET)  # Question 6 - E[T] vs Rho figures

	# M/G/1 Simulations for 0.35 <= Rho <= 0.95
	MG1 = Simulation('M', 'G', 1, inf, T, L, C, rho, L1, L2, prob)
	MG1.run_simulation()
	EN_MG1.append(MG1.EN)  # Question 3 - E[N] vs Rho figures
	Pidle_MG1.append(MG1.Pidle)  # Question 3 - Pidle vs Rho figures
	ET_MG1.append(MG1.ET)  # Question 6 - E[T] vs Rho figures


# Figures for E[N] and Pidle for the 3 queues
# E[N] vs. Rho figure - M/M/1 Queue
plt.figure()
plt.plot(rho_array, EN_MM1, marker='o')
plt.xlabel('Rho - Utilization factor')
plt.ylabel('E[N] - Average no. of packets in system (observer)')
plt.title('E[N] vs. Rho - M/M/1')

# E[N] vs. Rho figure - D/M/1 Queue
plt.figure()
plt.plot(rho_array, EN_DM1, marker='o')
plt.xlabel('Rho - Utilization factor')
plt.ylabel('E[N] - Average no. of packets in system (observer)')
plt.title('E[N] vs. Rho - D/M/1')

# E[N] vs. Rho figure - M/G/1 Queue
plt.figure()
plt.plot(rho_array, EN_MG1, marker='o')
plt.xlabel('Rho - Utilization factor')
plt.ylabel('E[N] - Average no. of packets in system (observer)')
plt.title('E[N] vs. Rho - M/G/1')

# E[N] vs. Rho - Comparative figure (M/M/1, D/M/1 and M/G/1)
plt.figure()
plt.plot(rho_array, EN_MM1, color = 'r', marker='x', label='M/M/1')
plt.plot(rho_array, EN_DM1, color = 'g', marker='x', label='D/M/1')
plt.plot(rho_array, EN_MG1, color = 'b', marker='x', label='M/G/1')
plt.xlabel('Rho - Utilization factor')
plt.ylabel('E[N] - Average no. of packets in system (observer)')
plt.title('E[N] vs. Rho - Comparison')
plt.legend()

# Output the values of E[N] graph in a text file
f.write('M/M/1 - E[N] (average no. of packets in system as seen by observer)\n{0}\n'.format(EN_MM1))
f.write('D/M/1 - E[N] (average no. of packets in system as seen by observer)\n{0}\n'.format(EN_DM1))
f.write('M/G/1 - E[N] (average no. of packets in system as seen by observer)\n{0}\n'.format(EN_MG1))


# Pidle vs. Rho figure - M/M/1 Queue
plt.figure()
plt.plot(rho_array, Pidle_MM1, marker='o')
plt.xlabel('Rho - Utilization factor')
plt.ylabel('P-idle - Proportion of time system is idle')
plt.title('P-idle vs. Rho - M/M/1')

# Pidle vs. Rho figure - D/M/1 Queue
plt.figure()
plt.plot(rho_array, Pidle_DM1, marker='o')
plt.xlabel('Rho - Utilization factor')
plt.ylabel('P-idle - Proportion of time system is idle')
plt.title('P-idle vs. Rho - D/M/1')

# Pidle vs. Rho figure - M/G/1 Queue 
plt.figure()
plt.plot(rho_array, Pidle_MG1, marker='o')
plt.xlabel('Rho - Utilization factor')
plt.ylabel('P-idle - Proportion of time system is idle')
plt.title('P-idle vs. Rho - M/G/1')

# Pidle vs. Rho - Comparative figure (M/M/1, D/M/1 and M/G/1)
plt.figure()
plt.plot(rho_array, Pidle_MM1, color = 'r', marker=2, linestyle = '-', label='M/M/1')
plt.plot(rho_array, Pidle_DM1, color = 'g', marker=3, linestyle = '--', label='D/M/1')
plt.plot(rho_array, Pidle_MG1, color = 'b', marker='x', linestyle = ':', label='M/G/1')
plt.xlabel('Rho - Utilization factor')
plt.ylabel('P-idle - Proportion of time system is idle')
plt.title('P-idle vs. Rho - Comparison')
plt.legend()

# Output the values of Pidle graph in a text file
f.write('M/M/1 - Pidle (proportion of time system is idle)\n{0}\n'.format(Pidle_MM1))
f.write('D/M/1 - Pidle (proportion of time system is idle)\n{0}\n'.format(Pidle_DM1))
f.write('M/G/1 - Pidle (proportion of time system is idle)\n{0}\n'.format(Pidle_MG1))




############   QUESTION 4 - Simulate M/M/1 with rho = 1.5   ############
print('\nQUESTION 4 - E[N] for M/M/1 when Rho = 1.5')
f.write('\nQUESTION 4 - E[N] for M/M/1 when Rho = 1.5\n')

rho = 1.5
# M/M/1 Simulaton for when rho=1.5
MM1_rho15 = Simulation('M', 'M', 1, inf, T, L, C, rho, L1, L2, prob)
MM1_rho15.run_simulation()

# D/M/1 Simulaton for when rho=1.5 (to compare with M/M/1)
DM1_rho15 = Simulation('D', 'M', 1, inf, T, L, C, rho, L1, L2, prob)
DM1_rho15.run_simulation()

# M/G/1 Simulaton for when rho=1.5 (to compare with M/M/1)
MG1_rho15 = Simulation('M', 'G', 1, inf, T, L, C, rho, L1, L2, prob)
MG1_rho15.run_simulation()

# M/M/1 Simulaton for when rho=0.5 (to compare with M/M/1 when rho=1.5)
rho = 0.5
MM1_rho05 = Simulation('M', 'M', 1, inf, T, L, C, rho, L1, L2, prob)
MM1_rho05.run_simulation()

# Output the E[N] of the simulations
print('\nM/M/1 - E[N] - Average no. of packets in system (rho = 1.5): {0:.1f}'.format(MM1_rho15.EN))
print('M/M/1 - Pidle - Proportion of time system is idle (rho = 1.5): {0:.1f}'.format(MM1_rho15.Pidle))
print('\nComparison of M/M/1 at rho=1.5 with other queues at rho=1.5:')
print('D/M/1 - E[N] - Average no. of packets in system (rho = 1.5): {0:.1f}'.format(DM1_rho15.EN))
print('D/M/1 - Pidle - Proportion of time system is idle (rho = 1.5): {0:.1f}'.format(DM1_rho15.Pidle))
print('M/G/1 - E[N] - Average no. of packets in system (rho = 1.5): {0:.1f}'.format(MG1_rho15.EN))
print('M/G/1 - Pidle - Proportion of time system is idle (rho = 1.5): {0:.1f}'.format(MG1_rho15.Pidle))
print('\nComparison of M/M/1 at rho=1.5 with M/M/1 at rho=0.5')
print('M/M/1 - E[N] - Average no. of packets in system (rho = 0.5): {0:.1f}'.format(MM1_rho05.EN))
print('M/M/1 - Pidle - Proportion of time system is idle (rho = 0.5): {0:.1f}'.format(MM1_rho05.Pidle))

f.write('M/M/1 - E[N] - Average no. of packets in system (rho = 1.5): {0:.1f}\n'.format(MM1_rho15.EN))
f.write('M/M/1 - Pidle - Proportion of time system is idle (rho = 1.5): {0:.1f}\n'.format(MM1_rho15.Pidle))
f.write('Comparison of M/M/1 with rho = 1.5 with other queues:\n')
f.write('D/M/1 - E[N] - Average no. of packets in system (rho = 1.5): {0:.1f}\n'.format(DM1_rho15.EN))
f.write('D/M/1 - Pidle - Proportion of time system is idle (rho = 1.5): {0:.1f}\n'.format(DM1_rho15.Pidle))
f.write('M/G/1 - E[N] - Average no. of packets in system (rho = 1.5): {0:.1f}\n'.format(MG1_rho15.EN))
f.write('M/G/1 - Pidle - Proportion of time system is idle (rho = 1.5): {0:.1f}\n'.format(MG1_rho15.Pidle))
f.write('Comparison of M/M/1 at rho=1.5 with M/M/1 at rho=0.5\n')
f.write('M/M/1 - E[N] - Average no. of packets in system (rho = 0.5): {0:.1f}\n'.format(MM1_rho05.EN))
f.write('M/M/1 - Pidle - Proportion of time system is idle (rho = 0.5): {0:.1f}\n'.format(MM1_rho05.Pidle))

# E[N] vs Observation Times figure for Rho=1.5
plt.figure()
plt.scatter(MM1_rho15.observation_times_list, MM1_rho15.Nt_observer, marker='.')
plt.xlabel('Observation Times (seconds)')
plt.ylabel('No. of packets in system')
plt.title('No. of packets vs. Observation Times  (Rho=1.5)')

# E[N] vs Observation Times figure for Rho=0.5 (to compare trends if any)
plt.figure()
plt.scatter(MM1_rho05.observation_times_list, MM1_rho05.Nt_observer, marker='.')
plt.xlabel('Observation Times (seconds)')
plt.ylabel('No. of packets in system')
plt.title('No. of packets vs. Observation Times  (Rho=0.5)')




############ QUESTION 5 - Compare E[N] and Ea[N] for M/M/1 and D/M/1 (uses values from Question 3)   ############
print('\nQUESTION 5 - Generating plots to compare E[N] and Ea[N] for M/M/1 and D/M/1')
f.write('\nQUESTION 5 - E[N] vs Ea[N] for M/M/1 and D/M/1\n')

# E[N] vs Ea[N] figure - M/M/1 Queue
plt.figure()
plt.plot(rho_array, EN_MM1, color = 'b', marker='x', linestyle = '-', label='Observer E[N]')
plt.plot(rho_array, EaN_MM1, color = 'g', marker='+', linestyle = '-', label='Arrival Ea[N]')
plt.xlabel('Rho - Utilization factor')
plt.ylabel('Average no. of packets in system')
plt.title('Average no. of packets vs. Rho - M/M/1')
plt.legend()

# Output the values of E[N] vs Ea[N] graph (M/M/1) in a text file
f.write('M/M/1 - E[N] (average no. of packets in system as seen by observer)\n{0}\n'.format(EN_MM1))
f.write('M/M/1 - Ea[N] (average no. of packets in system as seen by arrival)\n{0}\n'.format(EaN_MM1))


# E[N] vs Ea[N] figure - D/M/1 Queue
plt.figure()
plt.plot(rho_array, EN_DM1, color = 'b', marker='x', label='Observer E[N]')
plt.plot(rho_array, EaN_DM1, color = 'g', marker='+', label='Arrival Ea[N]')
plt.xlabel('Rho - Utilization factor')
plt.ylabel('Average no. of packets in system')
plt.title('Average no. of packets vs. Rho - D/M/1')
plt.legend()

# Output the values of E[N] vs Ea[N] graph (D/M/1) in a text file
f.write('D/M/1 - E[N] (average no. of packets in system as seen by observer)\n{0}\n'.format(EN_DM1))
f.write('D/M/1 - Ea[N] (average no. of packets in system as seen by arrival)\n{0}\n'.format(EaN_DM1))




############   QUESTION 6 - Compare E[T] (avg soujourn time) for the 3 queues uses values from Question 3) ############
print('\nQUESTION 6 - Generating plots to compare E[T] for the 3 queues')
f.write('\nQUESTION 6 - E[T] for the 3 queues\n')

# E[T] vs. Rho - Comparative figure (M/M/1, D/M/1 and M/G/1)
plt.figure()
plt.plot(rho_array, ET_MM1, color = 'r', marker='x', label='M/M/1')
plt.plot(rho_array, ET_DM1, color = 'g', marker='x', label='D/M/1')
plt.plot(rho_array, ET_MG1, color = 'b', marker='x', label='M/G/1')
plt.xlabel('Rho - Utilization factor')
plt.ylabel('E[T] - Average Soujorn Time (seconds)')
plt.title('E[T] vs. Rho')
plt.legend()

# Output the values of E[T] graph in a text file
f.write('M/M/1 - E[T] (average sojourn time in seconds)\n{0}\n'.format(ET_MM1))
f.write('D/M/1 - E[T] (average sojourn time in seconds)\n{0}\n'.format(ET_DM1))
f.write('M/G/1 - E[T] (average sojourn time in seconds)\n{0}\n'.format(ET_MG1))




############   QUESTION 7 - Build simulator for M/D/1/K   ############
# Changes to code done above in the create_event_scheduler() function 
# If else lines added at line 185 and 208




############   QUESTION 8 - P loss graphs for M/D/1/K   ############
print('\nQUESTION 8 -  Run simulation for M/D/1/K for rho from 0.4 to 3')
f.write('\nQUESTION 8 - M/D/1/K 0.4 <=Rho<= 2.0 (Step 0.1) and 2 <=Rho<= 3 (Step 0.2)\n')


# Empty lists to store values at each rho, to be used for plotting figures later
Ploss_10 = []  # Ploss values for M/D/1/K when K=10 
Ploss_50 = []  # Ploss values for M/D/1/K when K=50
Ploss_100 = [] # Ploss values for M/D/1/K when K=100 

rho_array = [] # List to store values of rho - to be used for x-axis when plotting graphs

# Loop for 0.4 <= Rho <= 3
rho = 0.4
while rho <= 3.001:
	rho_array.append(rho)

	# M/D/1/K Simulation when K=10
	MD1K_10 = Simulation('M', 'D', 1, 10, T, L, C, rho, L1, L2, prob)  # K = 10
	MD1K_10.run_simulation()
	Ploss_10.append(MD1K_10.Ploss)

	# M/D/1/K Simulation when K=50
	MD1K_50 = Simulation('M', 'D', 1, 50, T, L, C, rho, L1, L2, prob)  # K = 50
	MD1K_50.run_simulation()
	Ploss_50.append(MD1K_50.Ploss)

	# M/D/1/K Simulation when K=100
	MD1K_100 = Simulation('M', 'D', 1, 100, T, L, C, rho, L1, L2, prob)  # K = 100
	MD1K_100.run_simulation()
	Ploss_100.append(MD1K_100.Ploss)

	# Step size based on whether rho is less than 2 or greater
	if rho < 2:
		rho += 0.1
	else:
		rho += 0.2
	rho = round(rho, 1)


# Ploss vs. Rho - Comparative figure for M/D/1/K (K=10, K=50, and K=100)
plt.figure()
plt.plot(rho_array, Ploss_10, color = 'r', marker=2, linestyle = '-', label='K = 10')
plt.plot(rho_array, Ploss_50, color = 'g', marker=3, linestyle = '--', label='K = 50')
plt.plot(rho_array, Ploss_100, color = 'b', marker='x', linestyle = ':', label='K = 100')
plt.xlabel('Rho - Utilization factor')
plt.ylabel('P-loss - Proportion of packets lost')
plt.title('P-loss vs. Rho - M/D/1/K')
plt.legend()

# Output the values of Ploss graph in a text file
f.write('M/D/1/10 - Ploss (proportion of packets lost)\n{0}\n'.format(Ploss_10))
f.write('M/D/1/50 - Ploss (proportion of packets lost)\n{0}\n'.format(Ploss_50))
f.write('M/D/1/100 - Ploss (proportion of packets lost)\n{0}\n'.format(Ploss_100))




############   QUESTION 9 - M/D/2 vs. M/D/1 Queue   ############
print('\nQUESTION 9 - Compare M/D/2 (transmission rate C) with M/D/1 (transmission rate 2C)')
f.write('\nQUESTION 9 - M/D/2 and M/D/1 - 0.35 <= Rho <= 0.95 (Step size 0.05)\n')

C_double = 2*C  # Transmission rate of 2C for M/D/1

# Empty lists to store values at each rho, to be used for plotting figures later
EN_MD2_C = []  # E[N] values for M/D/2 with transmission rate C
EN_MD1_2C = [] # E[N] values for M/D/1 with transmission rate 2C

rho_array = [] # List to store values of rho - to be used for x-axis when plotting graphs

# Loop for 0.35 <= Rho <= 0.95 (Step size 0.05)
for a in np.arange(0.35, 1.0, 0.05):
	rho = round(a, 2)
	rho_array.append(rho)

	# M/D/2 with transmission rate C Simulation
	MD2_C = Simulation('M', 'D', 2, inf, T, L, C, rho, L1, L2, prob)
	MD2_C.run_simulation()
	EN_MD2_C.append(MD2_C.EN)

	# M/D/1 with transmission rate 2C Simulation
	MD1_2C = Simulation('M', 'D', 1, inf, T, L, C_double, rho, L1, L2, prob)
	MD1_2C.run_simulation()
	EN_MD1_2C.append(MD1_2C.EN)

# E[N] vs. Rho figure - Comparative figure (M/D/2 with rate C and M/D/1 with rate 2C)
plt.figure()
plt.plot(rho_array, EN_MD2_C, color = 'b', marker='x', linestyle = '-', label='M/D/2 - rate C')
plt.plot(rho_array, EN_MD1_2C, color = 'g', marker='+', linestyle = '--', label='M/D/1 - rate 2C')
plt.xlabel('Rho - Utilization factor')
plt.ylabel('E[N] - Average no. of packets in system (observer)')
plt.title('E[N] vs. Rho')
plt.legend()

# Output the values of E[N] graph for M/D/2 and M/D/1 in a text file
f.write('M/D/2 -E[N] (average no. of packets in system as seen by observer)\n{0}\n'.format(EN_MD2_C))
f.write('M/D/1 - E[N] (average no. of packets in system as seen by observer)\n{0}\n'.format(EN_MD1_2C))



# Close output file
f.closed

# Caclulate total time to run complete code
t1 = time.time()
print('\nTotal time taken for code to run: {0:.2f} mins'.format((t1-t0)/60))

# Show all figures at the end
plt.show()







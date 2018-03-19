# ECE610 Project - Queue Simulation
# Filza Mazahir - 20295951 - fmazahir@uwaterloo.ca


import random
import numpy as np
from math import log, inf
from collections import deque

import matplotlib as mpl 
mpl.use('TkAgg')
import matplotlib.pyplot as plt

import time



# Test uniform random values generator
def test_uniform_random_value_generator():
	# Generate 1000 uniform random values
	uniform_rv_list = []
	for x in range(1000):
		uniform_random_value = random.uniform(0, 1) # Python's built-in uniform random generator
		uniform_rv_list.append(uniform_random_value)

	# Check with mean and variance (convert to numpy array and use np.mean and np.var)
	uniform_rv_arr = np.array(uniform_rv_list)
	print('Mean - 1000 uniform random values (expected 0.5): {0:.3f}'.format(np.mean(uniform_rv_arr))) 
	print('Variance - 1000 uniform random values (expected 0.0833): {0:.3f}'.format(np.var(uniform_rv_arr))) 



# Generate Exponential Random Value - QUESTION 1
def generate_exponential_random_value(mu):
	uniform_rv = random.uniform(0, 1)
	exponential_random_value = -(log(uniform_rv))/mu
	return exponential_random_value
	# Python also has a random.expovariate(mu) function to generate exponential random values
	# But this function was written from the equation as per Question 1


# Generate 1000 Exponential Random Values to test - QUESTION 1 
def generate_1000_exponential_random_values():
	mu = 0.1    # 1/mu = 10seconds -> mu = 0.1
	exponential_rv_list = []
	for x in range(1000):
		expoential_rv = generate_exponential_random_value(mu)
		exponential_rv_list.append(expoential_rv)

	# Check with mean and variance 
	# Convert to numpy array and use np.mean and np.var
	exponential_rv_arr = np.array(exponential_rv_list)
	mean = np.mean(exponential_rv_arr)
	var = np.var(exponential_rv_arr)
	print('Mean - 1000 exponential random values (expected 10): {0:.1f}'.format(mean)) 
	print('Variance - 1000 exponential random values (expected 100): {0:.1f}'.format(var)) 
	return


class Simulation:
	
	def __init__(self, arrival_process, service_process, n, K, T, L, C, rho, L1, L2, prob):

		#Input variables
		self.arrival_process = arrival_process  # Arrival process given by 'M' or 'D'
		self.service_process = service_process  # Service process given by 'M', 'D' or 'G'
		self.n = n # Number of servers in the queue
		self.K = K  # Size of buffer

		self.T = T  # Total time to run the simulation
		self.L = L # Average packet length in bits
		self.C = C # Transmission rate in bits/second
		self.rho = rho # Utilization factor of the queue

		self.L1 = L1  # For General service process, L1 is the packet length with probability of prob
		self.L2 = L2 # For General service process, L2 is the packet length with probability of (1-prob)
		self.prob = prob  # For General service process, Probability of the packet to have L1 length

		self.lambd = (self.n * self.rho * self.C )/ self.L  # Arrival rate (average number of packets generated per second)
		self.alpha = self.lambd  # Observer rate - same as lambda so observations are in the same order as arrival packets

		# State of the system 
		self.Na = 0 # Number of packet arrival events
		self.Nd = 0 # Number of packet departure events
		self.No = 0 # Number of observervation events
		self.Nt = 0 # Total number of packets in the system

		# Helper variables to keep track of things and determine the output variables
		self.time_system_idle = 0  # Total time in the simulation that system is idle
		self.last_departure_time = 0  # Departure time of the last packet that departed from the system

		self.Tsojourn = []  # Total soujourn time for each packet
		self.Nt_observer = [] # No. of packets in system as seen by each observer event
		self.Nt_arrival = [] # No. of packets in system as seen by each arrival event

		self.dropped_packet_ids = set() # Set of packet ids for the packets that are dropped on arrival because of full buffer
		self.total_packets_generated = 0 # Total number of packets generated in the system
		
		print('Simulation -> {0}/{1}/{2}/{3} (Rho = {4}, Lambda = {5})'.format(self.arrival_process, self.service_process, self.n, self.K, self.rho, self.lambd))
		
		# Create Event Scheduler 
		self.create_event_scheduler()  


	# Function to generate arrival time based on the type of arrival process
	# Parameter is either alpha or lambda (based on whether the observer or arrival events are being generated)
	def generate_arrival_time(self, parameter, last_packet_arrival_time):
		# Poisson Distribution for arrival process
		if self.arrival_process == 'M':  
			random_time = generate_exponential_random_value(parameter)
			arrival_time = last_packet_arrival_time + random_time

		# Deterministic Distribution for arrival process (constant)
		elif self.arrival_process == 'D':  
			arrival_time = last_packet_arrival_time + (1/parameter)

		return(arrival_time)


	# Function to generate packet length based on type of service process
	def generate_packet_length(self):

		# Poisson Distribution for service process
		if self.service_process == 'M':
			packet_length = generate_exponential_random_value(1/self.L)

		# General Distribution with bipolar length for service process
		elif self.service_process == 'G':
			uniform_random_value = random.uniform(0, 1)
			if uniform_random_value <= self.prob:
				packet_length = self.L1
			else:
				packet_length = self.L2

		# Deterministic Distribution for service process (constant length)
		elif self.service_process == 'D':
			packet_length = self.L

		return(packet_length)


	# Create an Event Scheduler
	def create_event_scheduler(self):

		t0=time.time()

		# List of all events
		events_list = []

		# Generate set of random observation arrival times with parameter alpha
		observation_time = 0  # Observation time initialization, starts at t = 0
		while (observation_time < T):
			observation_time = self.generate_arrival_time(self.alpha, observation_time)
			event = (0, 'O', observation_time)
			events_list.append(event)

		# Helper variables needed to calculate arrival times, departure times, and sojourn time
		sojourn_time = 0  # Sojourn time taken for each packet
		arrival_time = 0  # Arrival time initialization, starts at t = 0
		server_available_time = [0 for i in range(self.n)] # List of available time for all n servers

		# Generate arrival time, packet length and packet id for the first packet in the Event Scheduler
		arrival_time = self.generate_arrival_time(self.lambd, arrival_time)
		packet_length = self.generate_packet_length()
		packet_id = 1
		self.total_packets_generated += 1
		
		# Calculate subsequent packet's arrival and depature times, and packet length
		while arrival_time < T:

			# Index of server that's available first (for 1 server queue, there is only 1 value here which is the minimum)
			i = server_available_time.index(min(server_available_time)) 

			# Calculate departure time and soujourn time
			# If server is free when the packet arrived (no wait)
			if server_available_time[i] <= arrival_time:
				departure_time = arrival_time + (packet_length/C)
				sojourn_time = packet_length/C	

			# If server is not free when the packet arrived (packet has to wait in the queue)
			else:
				departure_time = server_available_time[i] + (packet_length/C)
				sojourn_time = (server_available_time[i] - arrival_time)+ (packet_length/C)
				# print('Packet in buffer')

			# Server will be available when this packet departs
			server_available_time[i] = departure_time 

			# Add the arrival event and departure event to Event Scheduler
			events_list.append((packet_id, 'A', arrival_time))
			events_list.append((packet_id, 'D', departure_time))

			# Save the soujorn time of the packet
			self.Tsojourn.append(sojourn_time)

			# Calculate arrival time and packet length for the next packet
			arrival_time = self.generate_arrival_time(self.lambd, arrival_time)
			packet_length = self.generate_packet_length()
			packet_id += 1
			self.total_packets_generated += 1
		

		# Sort events list based on time after its completed
		events_list.sort(key=lambda event: event[2])

		# Create Event Scheduler as a double ended queue from the events_list
		self.Event_Scheduler = deque(events_list)

		t1 = time.time()
		if (t1-t0) > 6:
			print('Time taken in create event scheduler:', t1-t0)

		return


	# Functions to call at each observation event
	def observation_event_occured(self, current_time):
		self.No += 1 # Increment number of observation events

		# Record performance metric at the observation event
		self.Nt_observer.append(self.Nt)
		return

	# Functions to call at each arrival event
	def arrival_event_occured(self, current_time):

		# If there are no packets in the system currently (system is idle)
		if self.Nt == 0:
			self.time_system_idle += (current_time - self.last_departure_time)

		self.Na += 1  # Increment number of arrival events
		self.Nt = self.Na - self.Nd  # Update the current number of packets in the system

		# Record performance metric at packet arrival event
		self.Nt_arrival.append(self.Nt)

		return

	# Functions to call at each departure event
	def departure_event_occured(self, current_time):
		self.Nd += 1  # Increment number of departure events
		self.Nt = self.Na - self.Nd  # Update the current number of packets in the system

		self.last_departure_time = current_time # Update the departure time of the last packet that departed from system

		return

	# Run the simulation - goes through each event in the Event Scheduler and calls its respective function
	def run_simulation(self):

		t0 = time.time()


		while self.Event_Scheduler:
			# Dequeue the event from the Event Scheduler
			packet_id, event_type, event_time = self.Event_Scheduler.popleft() 

			# Observation Event
			if event_type == 'O':
				self.observation_event_occured(event_time)

			# Arrival Event
			elif event_type == 'A':
				# Check if buffer is full, then no event occured. Keep track of which packet was dropped
				if (self.Nt - self.n) >= self.K:
					self.dropped_packet_ids.add(packet_id) 

				# If buffer is not full, packet is not dropped, arrival event occurs
				else:
					self.arrival_event_occured(event_time)
					
			# Departure Event
			elif event_type == 'D':
				# Check if this packet was dropped earlier upon arrival, then ignore its corresponding departure event
				if packet_id in self.dropped_packet_ids: 
					continue 

				# Packet wasn't dropped - departure event occurs
				else:	
					self.departure_event_occured(event_time)


		# Calculate output variables after the simulation is run and complete
		self.Nt_observer_avg = sum(self.Nt_observer)/len(self.Nt_observer)
		self.Nt_arrival_avg = sum(self.Nt_arrival)/len(self.Nt_arrival)
		self.Tsojourn_avg = sum(self.Tsojourn)/len(self.Tsojourn)
		self.Pidle = self.time_system_idle/self.T * 100
		self.Ploss = len(self.dropped_packet_ids)/self.total_packets_generated * 100


		# print('Lambda', self.lambd)
		# print('Average no. of packets in system - observer', self.Nt_observer_avg)
		# print('Average no. of packets in system - arrival', self.Nt_arrival_avg)
		# print('Average Sojourn time', self.Tsojourn_avg)
		# print('P idle', self.Pidle, '%')
		# print('P loss', self.Ploss, '%')

		t1 = time.time()
		if (t1-t0) > 6:
			print('Time taken in run simulation:', t1-t0)

		return (self.Nt_observer_avg, self.Pidle, self.Nt_arrival_avg, self.Tsojourn_avg, self.Ploss)
		


########################################################################################
#### Main function - This is where all functions are called and simulations are run ####

t0=time.time()

# Question 1
print('\nQUESTION 1 - Exponential random variable generator')
test_uniform_random_value_generator()  #tests Python's built in uniform random generator
generate_1000_exponential_random_values() # uses-> generate_exponential_random_value(mu)


# Question 2 - M/M/1, D/M/1, and M/G/1 Queue Simulations 
print('\nQUESTION 2 - Build simulator for M/M/1, D/M/1 and M/G/1')

# Initialize all variables to run the simulations
T = 10000   
L = 20000 # Average length of packets
C = 2000000 # Transmission rate of output link in bits/sec
rho = 0.5  # Utilization factor of queue
L1 = 16000  # For M/G/1 
L2 = 21000   # For M/G/1
prob = 0.2   # For M/G/1

# Check if the value of T gives stable result
print('Checking if the T being used is generating a stable system')
print('T ->', T)

MM1 = Simulation('M', 'M', 1, inf, T, L, C, rho, L1, L2, prob)
No_of_packets_MM1, _, _, _, _ = MM1.run_simulation()

DM1 = Simulation('D', 'M', 1, inf, T, L, C, rho, L1, L2, prob)
No_of_packets_DM1, _, _, _, _ = DM1.run_simulation()

MG1 = Simulation('M', 'G', 1, inf, T, L, C, rho, L1, L2, prob)
No_of_packets_MG1, _, _, _, _ = MG1.run_simulation()

T_double = 2*T
print('2T ->', T_double)
MM1 = Simulation('M', 'M', 1, inf, T_double, L, C, rho, L1, L2, prob)
No_of_packets_MM1_2T, _, _, _, _ = MM1.run_simulation()

DM1 = Simulation('D', 'M', 1, inf, T_double, L, C, rho, L1, L2, prob)
No_of_packets_DM1_2T, _, _, _, _ = DM1.run_simulation()

MG1 = Simulation('M', 'G', 1, inf, T_double, L, C, rho, L1, L2, prob)
No_of_packets_MG1_2T, _, _, _, _ = MG1.run_simulation()

MM1_ratio = (No_of_packets_MM1 - No_of_packets_MM1_2T)/No_of_packets_MM1 * 100
DM1_ratio = (No_of_packets_DM1 - No_of_packets_DM1_2T)/No_of_packets_DM1 * 100
MG1_ratio = (No_of_packets_MG1 - No_of_packets_MG1_2T)/No_of_packets_MG1 * 100

print('Difference in values for M/M/1: {0:.1f}%'.format(MM1_ratio))
print('Difference in values for D/M1: {0:.1f}%'.format(DM1_ratio))
print('Difference in values for M/G/1: {0:.1f}%'.format(MG1_ratio))
if (MM1_ratio < 5 and DM1_ratio < 5 and MG1_ratio < 5):
	print('System is stable with T = {0} (values within 5%)'.format(T))
else:
	print('System NOT stable with T = {0} (values differ more than 5%)'.format(T))


# Question 3 - Run simulation for the 3 queues for rho from 0.35 to 0.95 and plot graphs
print('\nQUESTION 3 - Run simulation for the 3 queues for rho from 0.35 to 0.95')

EN_MM1 = []
Pidle_MM1 = []
EaN_MM1 = []
ET_MM1 = []

EN_DM1 = []
Pidle_DM1 = []
EaN_DM1 = []
ET_DM1 = []

EN_MG1 = []
Pidle_MG1 = []
ET_MG1 = []

rho_array = []
for a in np.arange(0.35, 1.0, 0.05):
	rho = round(a, 2)
	rho_array.append(rho)

	MM1 = Simulation('M', 'M', 1, inf, T, L, C, rho, L1, L2, prob)
	EN, Pidle, EaN, ET, _ = MM1.run_simulation()
	EN_MM1.append(EN)
	Pidle_MM1.append(Pidle)
	EaN_MM1.append(EaN)
	ET_MM1.append(ET)

	DM1 = Simulation('D', 'M', 1, inf, T, L, C, rho, L1, L2, prob)
	EN, Pidle, EaN, ET, _  = DM1.run_simulation()
	EN_DM1.append(EN)
	Pidle_DM1.append(Pidle)
	EaN_DM1.append(EaN)
	ET_DM1.append(ET)

	MG1 = Simulation('M', 'G', 1, inf, T, L, C, rho, L1, L2, prob)
	EN, Pidle, _ , ET, _ = MG1.run_simulation()
	EN_MG1.append(EN)
	Pidle_MG1.append(Pidle)
	ET_MG1.append(ET)

# Figures for the 3 queues
plt.figure()
plt.plot(rho_array, EN_MM1, marker='o')
plt.xlabel('Rho - Utilization factor')
plt.ylabel('E[N] - Average no. of packets in system (observer)')
plt.title('E[N] vs. Rho - M/M/1')

plt.figure()
plt.plot(rho_array, EN_DM1, marker='o')
plt.xlabel('Rho - Utilization factor')
plt.ylabel('E[N] - Average no. of packets in system (observer)')
plt.title('E[N] vs. Rho - D/M/1')

plt.figure()
plt.plot(rho_array, EN_MG1, marker='o')
plt.xlabel('Rho - Utilization factor')
plt.ylabel('E[N] - Average no. of packets in system (observer)')
plt.title('E[N] vs. Rho - M/G/1')

plt.figure()
plt.plot(rho_array, Pidle_MM1, marker='o')
plt.xlabel('Rho - Utilization factor')
plt.ylabel('P-idle - Proportion of time system is idle')
plt.title('P-idle vs. Rho - M/M/1')

plt.figure()
plt.plot(rho_array, Pidle_DM1, marker='o')
plt.xlabel('Rho - Utilization factor')
plt.ylabel('P-idle - Proportion of time system is idle')
plt.title('P-idle vs. Rho - D/M/1')

plt.figure()
plt.plot(rho_array, Pidle_MG1, marker='o')
plt.xlabel('Rho - Utilization factor')
plt.ylabel('P-idle - Proportion of time system is idle')
plt.title('P-idle vs. Rho - M/G/1')

# Comparitive figures (all queues in 1 figure)
plt.figure()
plt.plot(rho_array, EN_MM1, color = 'r', marker='x', label='M/M/1')
plt.plot(rho_array, EN_DM1, color = 'g', marker='x', label='D/M/1')
plt.plot(rho_array, EN_MG1, color = 'b', marker='x', label='M/G/1')
plt.xlabel('Rho - Utilization factor')
plt.ylabel('E[N] - Average no. of packets in system (observer)')
plt.title('E[N] vs. Rho - Comparison')
plt.legend()

plt.figure()
plt.plot(rho_array, Pidle_MM1, color = 'r', marker=2, linestyle = '-', label='M/M/1')
plt.plot(rho_array, Pidle_DM1, color = 'g', marker=3, linestyle = '--', label='D/M/1')
plt.plot(rho_array, Pidle_MG1, color = 'b', marker='x', linestyle = ':', label='M/G/1')
plt.xlabel('Rho - Utilization factor')
plt.ylabel('P-idle - Proportion of time system is idle')
plt.title('P-idle vs. Rho - Comparison')
plt.legend()


# Question 4 - Perhaps simulate all 3 queues with rho = 1.5
print('\nQUESTION 4 - Rho = 1.5 for all three queues')
rho = 1.5
MM1 = Simulation('M', 'M', 1, inf, T, L, C, rho, L1, L2, prob)
EN_MM1_rho, Pidle_MM1_rho, _ , _, _ = MM1.run_simulation()
DM1 = Simulation('D', 'M', 1, inf, T, L, C, rho, L1, L2, prob)
EN_DM1_rho, Pidle_DM1_rho, _ , _, _ = MM1.run_simulation()
MG1 = Simulation('M', 'G', 1, inf, T, L, C, rho, L1, L2, prob)
EN_MG1_rho, Pidle_MG1_rho, _ , _, _ = MM1.run_simulation()
print('Average no. of packets in system for M/M/1 (rho = 1.5): {0:1f}'.format(EN_MM1_rho))
print('Proportion of time system is idle for M/M/1 (rho = 1.5): {0:1f}'.format(Pidle_MM1_rho))
print('Average no. of packets in system for D/M/1 (rho = 1.5): {0:1f}'.format(EN_DM1_rho))
print('Proportion of time system is idle for D/M/1 (rho = 1.5): {0:1f}'.format(Pidle_DM1_rho))
print('Average no. of packets in system for M/G/1 (rho = 1.5): {0:1f}'.format(EN_MG1_rho))
print('Proportion of time system is idle for M/G/1 (rho = 1.5): {0:1f}'.format(Pidle_MG1_rho))


# Question 5 - Compare E[N] and Ea[N] for M/M/1 and D/M/1 (uses values from Question 3)
print('\nQUESTION 5 - Generating plots to compare E[N] and Ea[N] for M/M/1 and D/M/1')
plt.figure()
plt.plot(rho_array, EN_MM1, color = 'b', marker='x', label='Observer E[N]')
plt.plot(rho_array, EaN_MM1, color = 'g', marker='x', label='Arrival Ea[N]')
plt.xlabel('Rho - Utilization factor')
plt.ylabel('Average no. of packets in system')
plt.title('Average no. of packets vs. Rho - M/M/1')
plt.legend()

plt.figure()
plt.plot(rho_array, EN_DM1, color = 'b', marker='x', label='Observer E[N]')
plt.plot(rho_array, EaN_DM1, color = 'g', marker='x', label='Arrival Ea[N]')
plt.xlabel('Rho - Utilization factor')
plt.ylabel('Average no. of packets in system')
plt.title('Average no. of packets vs. Rho - D/M/1')
plt.legend()


# Question 6 - Compare E[T] (avg soujourn time) for the 3 queues uses values from Question 3)
print('\nQUESTION 6 - Generating plots to compare E[T] for the 3 queues')
plt.figure()
plt.plot(rho_array, ET_MM1, color = 'r', marker='x', label='M/M/1')
plt.plot(rho_array, ET_DM1, color = 'g', marker='x', label='D/M/1')
plt.plot(rho_array, ET_MG1, color = 'b', marker='x', label='M/G/1')
plt.xlabel('Rho - Utilization factor')
plt.ylabel('E[T] - Average Soujorn Time')
plt.title('E[T] vs. Rho')
plt.legend()


# Question 7 - M/D/1/K Simulation
print('\nQUESTION 7- Run simulation for M/D/1/K for rho from 0.4 to 3')

rho_array = []
Ploss_10 = []
Ploss_50 = []
Ploss_100 = []

rho = 0.4
while rho <= 3.001:
	rho_array.append(rho)

	MD1K_10 = Simulation('M', 'D', 1, 10, T, L, C, rho, L1, L2, prob)  # K = 10
	_, _, _, _, Ploss  = MD1K_10.run_simulation()
	Ploss_10.append(Ploss)

	MD1K_50 = Simulation('M', 'D', 1, 50, T, L, C, rho, L1, L2, prob)  # K = 50
	_, _, _, _, Ploss  = MD1K_50.run_simulation()
	Ploss_50.append(Ploss)

	MD1K_100 = Simulation('M', 'D', 1, 100, T, L, C, rho, L1, L2, prob)  # K = 100
	_, _, _, _, Ploss  = MD1K_100.run_simulation()
	Ploss_100.append(Ploss)

	if rho < 2:
		rho += 0.1
	else:
		rho += 0.2
	rho = round(rho, 1)

# Question 8 - P loss graphs
print('\nQUESTION 8 - Generating plots for P loss for M/D/1/K')

plt.figure()
plt.plot(rho_array, Ploss_10, color = 'r', marker=2, linestyle = '-', label='K = 10')
plt.plot(rho_array, Ploss_50, color = 'g', marker=3, linestyle = '--', label='K = 50')
plt.plot(rho_array, Ploss_100, color = 'b', marker='x', linestyle = ':', label='K = 100')
plt.xlabel('Rho - Utilization factor')
plt.ylabel('P-loss - Proportion of packets lost')
plt.title('P-loss vs. Rho - M/D/1/K')
plt.legend()


# Question 9 - M/D/2 Queue
print('\nQUESTION 9 - Compare M/D/2 (transmission rate C) with M/D/1 (transmission rate 2C)')

C_double = 2*C
EN_MD2_C = []
EN_MD1_2C = []
rho_array = []

for a in np.arange(0.35, 1.0, 0.05):
	rho = round(a, 2)
	rho_array.append(rho)

	MD2_C = Simulation('M', 'D', 2, inf, T, L, C, rho, L1, L2, prob)
	EN, _, _, _, _ = MD2_C.run_simulation()
	EN_MD2_C.append(EN)

	MD1_2C = Simulation('M', 'D', 1, inf, T, L, C_double, rho, L1, L2, prob)
	EN, _, _, _, _ = MD1_2C.run_simulation()
	EN_MD1_2C.append(EN)


plt.figure()
plt.plot(rho_array, EN_MD2_C, color = 'b', marker='x', linestyle = '-', label='M/D/2 - rate C')
plt.plot(rho_array, EN_MD1_2C, color = 'g', marker='+', linestyle = '--', label='M/D/1 - rate 2C')
plt.xlabel('Rho - Utilization factor')
plt.ylabel('E[N] - Average no. of packets in system (observer)')
plt.title('E[N] vs. Rho')
plt.legend()

t1 = time.time()
print('\nTotal time taken for code to run: {0:.2f} mins'.format((t1-t0)/60))

plt.show()







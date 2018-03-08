# ECE610 Project - Queue Simulation
# Filza Mazahir - 20295951 - fmazahir@uwaterloo.ca


import random
import numpy as np
from math import log



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

	# Check with mean and variance (convert to numpy array and use np.mean and np.var)
	exponential_rv_arr = np.array(exponential_rv_list)
	print('Mean - 1000 exponential random values (expected 10): {0:.1f}'.format(np.mean(exponential_rv_arr))) 
	print('Variance - 1000 exponential random values (expected 100): {0:.1f}'.format(np.var(exponential_rv_arr))) 





# Question 1
print('\nQUESTION 1')
test_uniform_random_value_generator();
generate_1000_exponential_random_values(); # uses-> generate_exponential_random_value(mu)


# Create an Event Scheduler

Event_Scheduler = [] # list of events, where events are stores as tuples ('type', time)
T = 10000
alpha = 0.1 # Parameter for observation times
lambd = 0.1
L = 100
C = 10

# Generate set of random observation times with parameter alpha
random_time = generate_exponential_random_value(alpha)
observation_time = random_time
while (observation_time < T):
	event = ('O', observation_time)
	Event_Scheduler.append(event)

	# Calculate observation time based on Poisson distribution with parameter alpha
	random_time = generate_exponential_random_value(alpha)
	observation_time += random_time

# Generate set of packet arrival times with parameter lambda and packet length with parameter 1/L

# Calculate first packet's arrival time and length
random_time = generate_exponential_random_value(lambd)
random_packet_length = generate_exponential_random_value(1/L)
arrival_time = random_time
while (arrival_time < T):
	# Add arrival event to ES
	event = ('A', arrival_time)
	Event_Scheduler.append(event)

	# Calculate departure time, and add departure event to ES
	departure_time = arrival_time + (L/C) # FIX THISSS!!!
	event = ('D', departure_time)
	Event_Scheduler.append(event)

	# Calculate arrival time based on Poisson distribution with parameter lambd
	random_time = generate_exponential_random_value(lambd)
	arrival_time += random_time

	# Calculate packet length based on exponential distribution with parameter 1/L
	random_packet_length = generate_exponential_random_value(1/L)


# Sort Event Scheduler list based on time
Event_Scheduler.sort(key=lambda tup: tup[1])


# Discrete Event Simulation for a simple queue with infinite buffer
Na = 0
Nd = 0
No = 0


# FIX THESEEE!!
def observation_event_occured():
	return

def arrival_event_occured():
	return

def departure_event_occured():
	return

while Event_Scheduler:
	if event[0] == 'O':
		observation_event_occured()
	elif event[0] == 'A':
		arrival_event_occured()
	elif event[0] == 'D':
		departure_event_occured()

	Event_Scheduler.pop(0) # Dequeue the event



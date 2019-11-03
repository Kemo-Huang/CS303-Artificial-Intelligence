# annealing
start_temperature = 1000
alpha = 0.99  # Temperature reduction multiplier
end_temperature = 0.0001
m0 = 50  # Times until next parameter update
before_timeout = 0.1

# path scanning
scanning_times = 10
discount = 1

# m0 * n = max_times
# start_temp * a^n = end_temp

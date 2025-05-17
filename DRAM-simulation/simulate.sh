./build/dramsim3main -c 100000 -o sim/stream  -t assignment_traces/dramsim3_stream_i10_n1000_rw2.trace  assignment_configs/LPDDR4_8Gb_x16_2400.ini
./build/dramsim3main -c 100000 -o sim/stream  -t assignment_traces/dramsim3_stream_i10_n1000_rw2.trace  assignment_configs/DDR4_8Gb_x8_3200.ini 
./build/dramsim3main -c 100000 -o sim/stream  -t assignment_traces/dramsim3_stream_i10_n1000_rw2.trace  assignment_configs/GDDR6_8Gb_x16.ini
./build/dramsim3main -c 100000 -o sim/stream  -t assignment_traces/dramsim3_stream_i10_n1000_rw2.trace  assignment_configs/HBM2_8Gb_x128.ini

./build/dramsim3main -c 100000 -o sim/random  -t assignment_traces/dramsim3_random_i10_n1000_rw2.trace assignment_configs/LPDDR4_8Gb_x16_2400.ini 
./build/dramsim3main -c 100000 -o sim/random  -t assignment_traces/dramsim3_random_i10_n1000_rw2.trace assignment_configs/DDR4_8Gb_x8_3200.ini
./build/dramsim3main -c 100000 -o sim/random  -t assignment_traces/dramsim3_random_i10_n1000_rw2.trace assignment_configs/GDDR6_8Gb_x16.ini
./build/dramsim3main -c 100000 -o sim/random  -t assignment_traces/dramsim3_random_i10_n1000_rw2.trace assignment_configs/HBM2_8Gb_x128.ini

./build/dramsim3main -c 100000 -o sim/mix  -t assignment_traces/dramsim3_mix_i10_n1000_rw2.trace assignment_configs/LPDDR4_8Gb_x16_2400.ini
./build/dramsim3main -c 100000 -o sim/mix  -t assignment_traces/dramsim3_mix_i10_n1000_rw2.trace assignment_configs/DDR4_8Gb_x8_3200.ini
./build/dramsim3main -c 100000 -o sim/mix  -t assignment_traces/dramsim3_mix_i10_n1000_rw2.trace assignment_configs/GDDR6_8Gb_x16.ini
./build/dramsim3main -c 100000 -o sim/mix  -t assignment_traces/dramsim3_mix_i10_n1000_rw2.trace assignment_configs/HBM2_8Gb_x128.ini
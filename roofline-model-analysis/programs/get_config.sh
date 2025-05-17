#!/bin/bash

get_system_configuration() {
    echo "Fetching system configuration..."

    # Number of sockets
    sockets=$(lscpu | grep 'Socket(s):' | awk '{print $2}')
    echo "Number of sockets: $sockets"

    # Number of cores per socket
    cores_per_socket=$(lscpu | grep 'Core(s) per socket:' | awk '{print $4}')
    echo "Cores per socket: $cores_per_socket"

    # Number of threads per core
    threads_per_core=$(lscpu | grep 'Thread(s) per core:' | awk '{print $4}')
    echo "Threads per core: $threads_per_core"

    # Total number of cores
    total_cores=$(lscpu | grep '^CPU(s):' | awk '{print $2}')
    echo "Total number of cores: $total_cores"

    # L1 Cache size
    #l1_cache=$(lscpu | grep 'L1d cache:' | awk '{print $3 $4}')
    l1d_cache=$(lscpu | grep 'L1d cache:')
    l1i_cache=$(lscpu | grep 'L1i cache:')
    echo "$l1d_cache"
    echo "$l1i_cache"

    # L2 Cache size
    #l2_cache=$(lscpu | grep 'L2 cache:' | awk '{print $3 $4}')
    l2_cache=$(lscpu | grep 'L2 cache:')
    echo "$l2_cache"

    # L3 Cache size
    #l3_cache=$(lscpu | grep 'L3 cache:' | awk '{print $3 $4}')
    l3_cache=$(lscpu | grep 'L3 cache:')
    echo "$l3_cache"

    # DRAM capacity (Total Memory)
    dram_capacity=$(grep MemTotal /proc/meminfo | awk '{print $2}')
    dram_capacity_gb=$(echo "scale=2; $dram_capacity / 1048576" | bc)
    echo "DRAM Capacity: $dram_capacity_gb GB"

    ## Optional: DRAM Bandwidth (if available via dmidecode or other tools)
    #if command -v dmidecode &> /dev/null; then
    #    dram_bandwidth=$(dmidecode --type 17 | grep -m 1 'Configured Clock Speed' | awk '{print $4 " MHz"}')
    #    echo "DRAM Bandwidth: $dram_bandwidth"
    #else
    #    echo "DRAM Bandwidth: Unable to retrieve (requires dmidecode)"
    #fi
}

# Execute the function
get_system_configuration

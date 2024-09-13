from numba import cuda
import numpy as np
import math
import matplotlib.pyplot as plt
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float32
from tqdm import tqdm

# CUDA Kernel for Monte Carlo Simulation
@cuda.jit(fastmath=True)
def monte_carlo_snowball_optimized(result, dt, rate, volatility, strike, knock_in, knock_out, rng_states, time_steps):
    tid = cuda.grid(1)  # Thread ID
    if tid < result.size:
        # Initialize asset price
        S = 100.0  
        
        # Barrier flags
        barrier_knock_in = False
        barrier_knock_out = False

        # Precompute drift and diffusion constants
        drift = (rate - 0.5 * volatility ** 2) * dt
        diffusion = volatility * math.sqrt(dt)

        # Simulate asset price path
        for _ in range(time_steps):
            # Generate two uniform random numbers
            u1 = xoroshiro128p_uniform_float32(rng_states, tid)
            u2 = xoroshiro128p_uniform_float32(rng_states, tid)
            
            # Box-Muller Transform to obtain a standard normal variable
            z = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
            
            # Update asset price
            S *= math.exp(drift + diffusion * z)

            # Check barrier conditions
            if not barrier_knock_in and S <= knock_in:
                barrier_knock_in = True
            if not barrier_knock_out and S >= knock_out:
                barrier_knock_out = True

        # Calculate payoff based on barrier conditions
        if barrier_knock_out:
            payoff = 0.0
        elif barrier_knock_in:
            payoff = S - strike if S > strike else 0.0
        else:
            payoff = 0.0  # Assuming payoff is 0 if neither barrier is hit

        # Store the result
        result[tid] = payoff

# Function to Price Snowball Option with Visualization
def price_snowball_option_with_visualization(num_paths, time_steps, T, rate, volatility, knock_in, knock_out, strike, batch_size=10000, num_sample_paths=100):
    dt = T / time_steps  # Time step size

    # Initialize result array
    results = np.zeros(num_paths, dtype=np.float32)

    # Initialize CURAND random number states
    rng_states = create_xoroshiro128p_states(num_paths, seed=42)

    # Define CUDA grid dimensions
    threads_per_block = 256
    blocks_per_grid = (num_paths + (threads_per_block - 1)) // threads_per_block

    # Transfer results array to device
    d_results = cuda.to_device(results)

    # Prepare for visualization
    plt.ion()  # Enable interactive mode
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Initialize plots
    # 1. Running Estimate of Option Price
    ax1.set_title("Running Estimate of Option Price")
    ax1.set_xlabel("Number of Simulations")
    ax1.set_ylabel("Estimated Option Price")
    running_estimates = []
    simulation_counts = []

    # 2. Histogram of Payoffs
    ax2.set_title("Histogram of Payoffs")
    ax2.set_xlabel("Payoff")
    ax2.set_ylabel("Frequency")
    histogram_bins = 50
    cumulative_payoffs = []

    # Variables to track progress
    total_batches = math.ceil(num_paths / batch_size)
    cumulative_sum = 0.0
    cumulative_count = 0

    # Use tqdm for progress bar
    with tqdm(total=num_paths, desc="Simulations") as pbar:
        for batch in range(total_batches):
            # Determine the number of simulations in this batch
            current_batch_size = min(batch_size, num_paths - batch * batch_size)

            # Launch the CUDA kernel
            monte_carlo_snowball_optimized[blocks_per_grid, threads_per_block](
                d_results, dt, rate, volatility, strike, knock_in, knock_out, rng_states, time_steps
            )

            # Copy results back to host
            d_results.copy_to_host(results)

            # Update cumulative sum and count
            batch_results = results[batch * batch_size : (batch + 1) * batch_size]
            cumulative_sum += np.sum(batch_results)
            cumulative_count += current_batch_size

            # Calculate running estimate
            running_mean = cumulative_sum / cumulative_count
            running_estimates.append(running_mean)
            simulation_counts.append(cumulative_count)

            # Update histogram data
            cumulative_payoffs.extend(batch_results)

            # Update plots every batch
            ax1.clear()
            ax1.set_title("Running Estimate of Option Price")
            ax1.set_xlabel("Number of Simulations")
            ax1.set_ylabel("Estimated Option Price")
            ax1.plot(simulation_counts, running_estimates, color='blue')

            ax2.clear()
            ax2.set_title("Histogram of Payoffs")
            ax2.set_xlabel("Payoff")
            ax2.set_ylabel("Frequency")
            ax2.hist(cumulative_payoffs, bins=histogram_bins, color='green', alpha=0.7)

            plt.tight_layout()
            plt.pause(0.01)  # Brief pause to update the plots

            # Update progress bar
            pbar.update(current_batch_size)

    plt.ioff()  # Disable interactive mode
    plt.savefig("snowball_option_simulation.png")

    # Final Option Price Calculation
    option_price = running_mean * math.exp(-rate * T)
    return option_price

# Main Execution with Sample Asset Paths Visualization
def main():
    # Simulation Parameters
    num_simulations = 1000000       # Total number of Monte Carlo simulations
    time_steps = 365               # Number of time steps (e.g., daily steps for 1 year)
    T = 1.0                        # Total time in years
    risk_free_rate = 0.05          # Risk-free interest rate
    volatility = 0.2               # Asset volatility
    strike = 100.0                 # Strike price
    knock_in_barrier = 90.0        # Knock-in barrier
    knock_out_barrier = 110.0      # Knock-out barrier
    batch_size = 10000             # Number of simulations per batch for visualization

    # Number of sample paths to visualize
    num_sample_paths = 100

    # Price the Snowball Option with Visualization
    option_price = price_snowball_option_with_visualization(
        num_paths=num_simulations, 
        time_steps=time_steps, 
        T=T,
        rate=risk_free_rate, 
        volatility=volatility, 
        knock_in=knock_in_barrier, 
        knock_out=knock_out_barrier, 
        strike=strike,
        batch_size=batch_size,
        num_sample_paths=num_sample_paths
    )

    # Output the Final Option Price
    print(f"\nFinal Snowball Option Price: {option_price:.4f}")

if __name__ == "__main__":
    main()

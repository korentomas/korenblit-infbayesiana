using CSV
using DataFrames
using Distributions
using Plots
using Dates

# --- Parameters ---
INITIAL_BANKROLL = 1.0
FRACTIONAL_KELLY = 0.5
BETA = 1.0

# --- Functions ---

# Win probability using Gaussian CDF
function predict_win_prob(mu1, sigma1, mu2, sigma2, beta)
    mu_diff = mu1 - mu2
    sigma_diff = sqrt(sigma1^2 + sigma2^2 + 2 * beta^2)
    return cdf(Normal(0, 1), mu_diff / sigma_diff)
end

# Kelly fraction
function kelly_fraction(p, odds; fractional_kelly=FRACTIONAL_KELLY)
    if odds < 1.01 || p <= 0.0 || p >= 1.0
        return 0.0
    end
    b = odds - 1.0
    edge = p * (b + 1) - 1
    return max(0.0, fractional_kelly * edge / b)
end

# --- Main simulation ---
function simulate_kelly(df)
    bankroll = INITIAL_BANKROLL
    bankroll_history = Float64[]
    bet_outcomes = String[]  # Track bet decisions
    
    for row in eachrow(df)
        # Skip if critical columns are missing
        if ismissing(row.b_winner) || ismissing(row.b_loser) || ismissing(row.winner) || ismissing(row.loser)
            push!(bankroll_history, bankroll)
            push!(bet_outcomes, "skip")
            continue
        end

        # Extract parameters
        mu_w = row.m_winner
        sigma_w = row.s_winner
        mu_l = row.m_loser
        sigma_l = row.s_loser
        odds_winner = row.b_winner
        odds_loser = row.b_loser

        # Calculate win probabilities
        p_win = predict_win_prob(mu_w, sigma_w, mu_l, sigma_l, BETA)
        p_lose = 1 - p_win

        # Calculate Kelly fractions
        f_winner = kelly_fraction(p_win, odds_winner)
        f_loser = kelly_fraction(p_lose, odds_loser)

        # Determine which bet to make (if any)
        if f_winner > f_loser && f_winner > 0
            # Bet on winner
            stake = bankroll * f_winner
            if row.winner == row.winner  # This compares the winner name
                bankroll += stake * (odds_winner - 1)
                push!(bet_outcomes, "win (winner)")
            else
                bankroll -= stake
                push!(bet_outcomes, "loss (winner)")
            end
        elseif f_loser > 0
            # Bet on loser
            stake = bankroll * f_loser
            if row.winner == row.loser  # Compare against loser name
                bankroll += stake * (odds_loser - 1)
                push!(bet_outcomes, "win (loser)")
            else
                bankroll -= stake
                push!(bet_outcomes, "loss (loser)")
            end
        else
            # No positive Kelly fraction found
            push!(bankroll_history, bankroll)
            push!(bet_outcomes, "no bet")
            continue
        end

        # Record bankroll and check bankruptcy
        push!(bankroll_history, bankroll)
        if bankroll < 1e-5
            println("Bankrupt at row $(row.date)")
            break
        end
    end
    
    return bankroll_history, bet_outcomes
end

# --- Load data and run simulation ---
df = CSV.read("predicciones_atp.csv", DataFrame)

# Check required columns using proper method
required_cols = ["b_winner", "b_loser", "winner", "loser", "m_winner", "s_winner", "m_loser", "s_loser", "date"]
missing_cols = [col for col in required_cols if !(Symbol(col) in propertynames(df))]

if !isempty(missing_cols)
    error("Missing required columns: " * join(missing_cols, ", "))
end

# Run simulation
bankroll_history, bet_outcomes = simulate_kelly(df)
df.bankroll = bankroll_history
df.bet_decision = bet_outcomes

# Save results
CSV.write("predicciones_atp_con_kelly.csv", df)

# Plot bankroll growth
plot(df.date, df.bankroll, 
     xlabel="Date", ylabel="Bankroll", 
     title="Kelly Criterion Bankroll Growth", 
     legend=false, linewidth=2)
savefig("bankroll_growth_julia.png")
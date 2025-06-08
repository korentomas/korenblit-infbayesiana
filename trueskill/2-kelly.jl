using CSV
using DataFrames
using Distributions
using Plots

# --- Parameters ---
INITIAL_BANKROLL = 1.0
FRACTIONAL_KELLY = 0.5
BETA = 1.0

# --- Functions ---

# Win probability using Gaussian CDF
function predict_win_prob(mu1, sigma1, mu2, sigma2, beta)
    mu_diff = mu1 - mu2
    sigma_diff = sqrt(sigma1^2 + sigma2^2 + 2 * beta^2)
    return cdf(Normal(0,1), mu_diff / sigma_diff)
end

# Kelly fraction
function kelly_fraction(p, odds; fractional_kelly=FRACTIONAL_KELLY)
    if odds < 1.01
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
    for row in eachrow(df)
        mu_w = row.m_winner
        sigma_w = row.s_winner
        mu_l = row.m_loser
        sigma_l = row.s_loser
        odds = row.b_winner

        # Skip if odds are missing
        if ismissing(odds)
            push!(bankroll_history, bankroll)
            continue
        end

        p_win = predict_win_prob(mu_w, sigma_w, mu_l, sigma_l, BETA)
        f = kelly_fraction(p_win, odds)
        stake = bankroll * f

        # Simulate bet: always bet on the winner (since we know the result)
        bankroll -= stake
        bankroll += stake * odds

        push!(bankroll_history, bankroll)
        if bankroll < 1
            println("Bankrupt at match $(row)")
            break
        end
    end
    return bankroll_history
end

df = CSV.read("predicciones_atp.csv", DataFrame)
df.bankroll = simulate_kelly(df)
CSV.write("predicciones_atp_con_kelly.csv", df)

# Plot
plot(df.date, df.bankroll, xlabel="Date", ylabel="Bankroll", title="Bankroll Growth Over Time", legend=false)
savefig("bankroll_growth_julia.png")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime



class StockPortfolioEnv():
    metadata = {'render.modes': ['human']}

    def __init__(self,
                 df,
                 stock_dim,
                 hmax,
                 initial_amount,
                 transaction_cost_amount,
                 reward_scaling,
                 state_space,
                 action_space,
                 tech_indicator_list,
                 turbulence_threshold=None,
                 lookback=5,
                 min=0):
        # Initialization of environment variables
        self.min = min  # Current time index
        self.lookback = lookback  # Number of previous steps for state construction
        self.df = df  # Market data
        self.stock_dim = stock_dim  # Number of stocks
        self.hmax = hmax  # Max shares per transaction
        self.initial_amount = initial_amount  # Starting portfolio cash value
        self.transaction_cost_amount = transaction_cost_amount  # Cost per share transaction
        self.reward_scaling = reward_scaling  # Scaling factor for reward
        self.state_space = state_space  # Dimensions of state space
        self.action_space = 0  # Long/Short/StopLoss/TakeProfit
        self.tech_indicator_list = tech_indicator_list  # List of technical indicators

        # Observation space includes technical indicators, trade history, and predictions
        self.observation_space = 0

        # Initialize data, state, and environment variables
        self.current_timestamp = self.df['timestamp'].sort_values().unique()[self.min]
        self.data = self.df[self.df['timestamp'] == self.current_timestamp]
        self.covs = self.data['cov_list'].values[0]
        self.state = self._construct_state()
        self.terminal = False
        self.turbulence_threshold = turbulence_threshold
        self.portfolio_value = self.initial_amount
        self.cash = self.initial_amount

        # Memory for logging and tracking
        self.asset_memory = [self.initial_amount]
        self.portfolio_return_memory = [0]
        self.actions_memory = [[0] * self.stock_dim]
        self.date_memory = [current_timestamp]
        self.share_holdings = np.zeros(self.stock_dim)  # Long/Short positions for each stock

    def _construct_state(self):
        """Construct the current state with historical data, trades, and predictions."""
        historical_data = self.df.loc[max(0, self.min - self.lookback):self.min]
        historical_tech = [historical_data[tech].values.tolist() for tech in self.tech_indicator_list]

        # Padding if historical data is less than lookback
        if len(historical_data) < self.lookback:
            padding = [np.zeros(self.stock_dim) for _ in range(self.lookback - len(historical_data))]
            historical_tech = padding + historical_tech

        # Include trade and prediction history
        historical_trade = historical_data['trade_list'].values.tolist()
        historical_prediction = historical_data['prediction_list'].values.tolist()

        if len(historical_trade) < self.lookback:
            trade_padding = [np.zeros(self.stock_dim) for _ in range(self.lookback - len(historical_trade))]
            historical_trade = trade_padding + historical_trade

        if len(historical_prediction) < self.lookback:
            prediction_padding = [np.zeros(self.stock_dim) for _ in range(self.lookback - len(historical_prediction))]
            historical_prediction = prediction_padding + historical_prediction

        # Append data to form the state
        state = np.append(
            np.array(self.covs),
            historical_tech + historical_trade + historical_prediction,
            axis=0
        )
        return state

    def _sell_stock(self, stock_index, sell_amount, current_price):
        """Sell stock, including handling short positions."""
        sell_value = sell_amount * current_price
        transaction_cost = sell_amount * self.transaction_cost_amount
        sell_value -= transaction_cost
        self.cash += sell_value
        self.portfolio_value -= transaction_cost
        self.share_holdings[stock_index] -= sell_amount  # Update holdings (allows negative for shorts)
        self.actions_memory[-1][stock_index] -= sell_amount

    def _buy_stock(self, stock_index, buy_amount, current_price):
        """Buy stock, including closing short positions."""
        buy_value = buy_amount * current_price
        transaction_cost = buy_amount * self.transaction_cost_amount
        buy_value += transaction_cost
        self.cash -= buy_value
        self.portfolio_value -= transaction_cost
        self.share_holdings[stock_index] += buy_amount  # Adjust holdings for both long and short
        self.actions_memory[-1][stock_index] += buy_amount

    def _sell_all_stocks(self):
        """Sell all long positions and buy back all short positions."""
        for i in range(self.stock_dim):
            current_price = self.data['close'].values[i]
            if self.share_holdings[i] > 0:  # Long positions
                self._sell_stock(i, self.share_holdings[i], current_price)
            elif self.share_holdings[i] < 0:  # Short positions
                self._buy_stock(i, -self.share_holdings[i], current_price)  # Buy back short shares

    def step(self, actions):
        """Execute a single step in the environment."""
        self.terminal = self.min >= len(self.df.timestamp.unique()) - 1
        last_minute_of_day = self.current_timestamp == self.df[self.df['timestamp'].dt.date == self.data.timestamp.dt.date[0]]['timestamp'].max()

        if self.terminal:
            self._sell_all_stocks()

            # Log cumulative rewards
            df = pd.DataFrame(self.portfolio_return_memory)
            df.columns = ['minutely_return']
            plt.plot(df.minutely_return.cumsum(), 'r')
            plt.savefig('results/cumulative_reward.png')
            plt.close()

            plt.plot(self.portfolio_return_memory, 'r')
            plt.savefig('results/rewards.png')
            plt.close()

            # Calculate and save Sharpe ratio
            total_days = pd.to_datetime(self.df['timestamp']).dt.date.nunique()
            risk_free_rate = 0.04
            scaling_factor = 390 * total_days
            mean_return_annualized = df['minutely_return'].mean() * scaling_factor
            std_return_annualized = df['minutely_return'].std() * (scaling_factor ** 0.5)
            sharpe = (mean_return_annualized - risk_free_rate) / std_return_annualized

            with open('results/sharpe_ratio.txt', 'w') as f:
                f.write(f'Sharpe Ratio: {sharpe}\n')

            return self.state, self.reward, self.terminal, {}

        else:
            if last_minute_of_day:
                new_weights = np.zeros_like(actions[:, 0])  # Set all weights to zero
            else:
                new_weights = self.softmax_normalization(actions[:, 0])

            stop_loss = actions[:, 1]
            take_profit = actions[:, 2]

            # Execute trades based on weight differences
            previous_weights = np.array(self.actions_memory[-1][:, 0])
            weight_diff = new_weights - previous_weights

            for i, diff in enumerate(weight_diff):
                current_price = self.data['close'].values[i]
                if diff > 0:  # Increase weight: Buy shares
                    buy_amount = int(diff * self.portfolio_value / current_price)
                    self._buy_stock(i, buy_amount, current_price)
                elif diff < 0:  # Decrease weight: Sell shares
                    sell_amount = int(-diff * self.portfolio_value / current_price)
                    self._sell_stock(i, sell_amount, current_price)

            self.actions_memory.append(np.column_stack((new_weights, stop_loss, take_profit)))
            last_day_memory = self.data

            # Update state and portfolio values
            self.min += 1
            self.data = self.df.loc[self.min, :]
            self.covs = self.data['cov_list'].values[0]
            self.state = self._construct_state()

            portfolio_return, updated_weights = self.calculate_portfolio_return(
                last_day_memory, new_weights, stop_loss, take_profit)

            self.actions_memory[-1][:, 0] = updated_weights
            self.portfolio_return_memory.append(portfolio_return)
            self.date_memory.append(self.data.timestamp.unique()[0])
            self.asset_memory.append(self.portfolio_value)

            self.reward = self.portfolio_value

        return self.state, self.reward, self.terminal, {}

    def calculate_portfolio_return(self, last_day, weights, stop_loss, take_profit):
        """Calculate returns for the portfolio, including short positions."""
        price_change = (self.data['close'].values - last_day['close'].values)
        updated_weights = weights.copy()
        returns = []

        for i, weight in enumerate(weights):
            low = self.data['low'].values[i]
            high = self.data['high'].values[i]
            close_price = self.data['close'].values[i]

            stop_loss_price = last_day['close'].values[i] * (1 - stop_loss[i])
            take_profit_price = last_day['close'].values[i] * (1 + take_profit[i])

            if low <= stop_loss_price and self.share_holdings[i] < 0:  # Short stop-loss
                current_return = (stop_loss_price - last_day['close'].values[i]) * self.share_holdings[i]
                transaction_cost = abs(self.share_holdings[i]) * self.transaction_cost_amount
                current_return -= transaction_cost
                self.share_holdings[i] = 0  # Exit short position
                self.cash += current_return
            elif high >= take_profit_price and self.share_holdings[i] < 0:  # Short take-profit
                current_return = (take_profit_price - last_day['close'].values[i]) * self.share_holdings[i]
                transaction_cost = abs(self.share_holdings[i]) * self.transaction_cost_amount
                current_return -= transaction_cost
                self.share_holdings[i] = 0  # Exit short position
                self.cash += current_return
            else:
                current_return = price_change[i] * self.share_holdings[i]

            returns.append(current_return)

        portfolio_return = np.sum(returns)
        self.portfolio_value += portfolio_return

        # Recalculate weights for portfolio
        for i, share in enumerate(self.share_holdings):
            updated_weights[i] = share * self.data['close'].values[i] / max(self.portfolio_value, 1e-10)

        return portfolio_return, updated_weights

    def reset(self):
        """Reset the environment to its initial state."""
        self.asset_memory = [self.initial_amount]
        self.min = 0
        self.current_timestamp = self.df['timestamp'].sort_values().unique()[self.min]
        self.data = self.df[self.df['timestamp'] == self.current_timestamp]
        self.covs = self.data['cov_list'].values
        self.state = self._construct_state()
        self.portfolio_value = self.initial_amount
        self.cash = self.initial_amount
        self.terminal = False
        self.portfolio_return_memory = [0]
        self.actions_memory = [[0] * self.stock_dim]
        self.date_memory = [self.data.timestamp.unique()[0]]
        self.share_holdings = np.zeros(self.stock_dim)
        return self.state

    def render(self, mode='human'):
        """Render the current environment state."""
        return self.state

    def softmax_normalization(self, actions):
        """Normalize actions for valid weights."""
        numerator = np.exp(actions)
        denominator = np.sum(np.exp(actions))
        softmax_output = numerator / denominator
        return softmax_output

    def save_asset_memory(self):
        """Save portfolio values over time."""
        date_list = self.date_memory
        portfolio_return = self.portfolio_return_memory
        df_account_value = pd.DataFrame({'date': date_list, 'minutely_return': portfolio_return})
        return df_account_value

    def save_action_memory(self):
        """Save actions taken over time."""
        date_list = self.date_memory
        df_date = pd.DataFrame(date_list)
        df_date.columns = ['timestamp']

        action_list = self.actions_memory
        df_actions = pd.DataFrame(action_list)
        df_actions.columns = self.data.tic.values
        df_actions.index = df_date.timestamp
        return df_actions

    def _seed(self, seed=None):
        """Set random seed for reproducibility."""
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def get_sb_env(self):
        """Get the stable-baselines environment."""
        e = 0 #DummyVecEnv([lambda: self])
        obs = e.reset()
        return e, obs


# Define timestamps for 10 time steps
# Define timestamps as datetime objects for 10 time steps
timestamps = [datetime.strptime(f'2024-01-01 09:30:{str(i).zfill(2)}', '%Y-%m-%d %H:%M:%S') for i in range(10)]


# Define the tickers
tickers = ['AAPL', 'MSFT', 'GOOG']

# Generate repeated data for each ticker at each timestamp
data = []
for timestamp in timestamps:
    for tic in tickers:
        data.append({
            'timestamp': timestamp,
            'tic': tic,
            'open': np.random.uniform(100, 200),  # Random open price
            'high': np.random.uniform(200, 300),  # Random high price
            'low': np.random.uniform(50, 100),   # Random low price
            'close': np.random.uniform(100, 200),  # Random close price
            'volume': np.random.randint(1000, 10000),  # Random volume
            'trade_list': [0, 0, 0],  # Placeholder trade data
            'prediction_list': [0.3, 0.5, 0.2],  # Placeholder predictions
            'cov_list': np.eye(len(tickers)).tolist()  # Covariance matrix
        })

# Generate repeated action data for each ticker at each timestamp
actions_data = []
for timestamp in timestamps:
    for tic in tickers:
        actions_data.append({
            'timestamp': timestamp,
            'tic': tic,
            'weight': np.random.uniform(-1, 1),  # Random portfolio weight (allows shorts)
            'stop_loss': np.random.uniform(0.01, 0.1),  # Random stop-loss percentage
            'take_profit': np.random.uniform(0.1, 0.2)  # Random take-profit percentage
        })

# Create a DataFrame from the list of dictionaries
df_actions = pd.DataFrame(actions_data)

# Create a DataFrame from the list of dictionaries
train = pd.DataFrame(data)



stock_dimension = len(train.tic.unique())
state_space = stock_dimension
env_kwargs = {
    "hmax": 100,
    "initial_amount": 1000000,
    "transaction_cost_amount": 0.0035,
    "state_space": state_space,
    "stock_dim": stock_dimension,
    "tech_indicator_list": [],
    "action_space": stock_dimension,
    "reward_scaling": 1e-4
}

e_train = StockPortfolioEnv(df = train, **env_kwargs)

for i in range(10):

    # Filter actions for the i timestamp
    second_timestamp = df_actions['timestamp'].sort_values().unique()[i]  # Get the second timestamp
    second_actions = df_actions[df_actions['timestamp'] == second_timestamp]

    # Prepare the actions as a NumPy array for the environment
    actions = second_actions[['weight', 'stop_loss', 'take_profit']].to_numpy()

    # Execute the second step
    state, reward, terminal, info = e_train.step(actions)

    # Print the results
    print("State after second step:")
    print(state)
    print("\nReward after second step:", reward)
    print("\nTerminal status:", terminal)
    print("\nAdditional info:", info)









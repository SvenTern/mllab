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
                 lookback=5):
        # Initialization of environment variables
        self.min = 0  # Current time index
        self.lookback = lookback  # Number of previous steps for state construction
        self.df = df  # Market data
        self.stock_dim = stock_dim  # Number of stocks
        self.hmax = hmax  # Max shares per transaction
        self.initial_amount = initial_amount  # Starting portfolio cash value
        self.transaction_cost_amount = transaction_cost_amount  # Cost per share transaction
        self.reward_scaling = reward_scaling  # Scaling factor for reward
        self.state_space = state_space  # Dimensions of state space
        self.action_space = 0 #spaces.Box(low=-1, high=1, shape=(action_space, 3))  # Long/Short/StopLoss/TakeProfit
        self.tech_indicator_list = tech_indicator_list  # List of technical indicators

        # Precompute timestamps and data mapping
        self.timestamps = df['timestamp'].sort_values().unique()
        self.data_map = {ts: df[df['timestamp'] == ts] for ts in self.timestamps}
        self.state = self._construct_state()
        self.terminal = False
        self.portfolio_value = self.initial_amount
        self.cash = self.initial_amount

        # Memory for tracking and logging
        self.asset_memory = [self.initial_amount]
        self.portfolio_return_memory = [0]
        self.actions_memory = [[[0]] * self.stock_dim]
        self.date_memory = [self.timestamps[0]]
        self.share_holdings = np.zeros(self.stock_dim)  # Long/Short positions for each stock

    def _construct_state(self):
        """Construct the current state with historical data."""
        start_index = max(0, self.min - self.lookback + 1)
        historical_data = self.df.iloc[start_index:self.min + 1]

        # Collect technical indicators
        state_data = []
        for tech in self.tech_indicator_list:
            tech_values = historical_data[tech].values[-self.lookback:] if len(historical_data) >= self.lookback else np.zeros(self.lookback)
            state_data.append(tech_values)

        return np.concatenate(state_data)

    def _sell_stock(self, stock_index, sell_amount, current_price):
        """Sell stock and handle cash and holdings."""
        sell_value = sell_amount * current_price
        transaction_cost = abs(sell_amount) * self.transaction_cost_amount
        self.cash += sell_value - transaction_cost
        self.share_holdings[stock_index] -= sell_amount

    def _buy_stock(self, stock_index, buy_amount, current_price):
        """Buy stock and handle cash and holdings."""
        buy_value = buy_amount * current_price
        transaction_cost = abs(buy_amount) * self.transaction_cost_amount
        self.cash -= buy_value - transaction_cost
        self.share_holdings[stock_index] += buy_amount

    def _sell_all_stocks(self):
        """Sell all long and short positions."""
        for i, holding in enumerate(self.share_holdings):
            current_price = self.data_map[self.timestamps[self.min]]['close'].values[i]
            if holding > 0:
                self._sell_stock(i, holding, current_price)
            elif holding < 0:
                self._buy_stock(i, -holding, current_price)

    def step(self, actions):
        """Execute one step in the environment."""
        self.terminal = self.min >= len(self.timestamps) - 1
        last_minute_of_day = self.min < len(self.timestamps) - 1 and self.timestamps[self.min].date() != self.timestamps[self.min + 1].date()

        if self.terminal:
            self._sell_all_stocks()
            df = pd.DataFrame(self.portfolio_return_memory, columns=['return'])
            plt.plot(df['return'].cumsum())
            plt.savefig('cumulative_reward.png')
            plt.close()

            return self.state, self.reward, self.terminal, {}

        # Normalize weights for non-terminal, non-last-minute steps
        new_weights = np.zeros_like(actions[:, 0]) if last_minute_of_day else self.softmax_normalization(actions[:, 0])
        stop_loss = actions[:, 1]
        take_profit = actions[:, 2]
        weight_diff = new_weights - np.array(self.actions_memory[-1])

        for i, diff in enumerate(weight_diff):
            current_price = self.data_map[self.timestamps[self.min]]['close'].values[i]
            if diff > 0:
                self._buy_stock(i, int(diff * self.portfolio_value / current_price), current_price)
            elif diff < 0:
                self._sell_stock(i, int(-diff * self.portfolio_value / current_price), current_price)

        self.actions_memory.append(np.column_stack((new_weights, stop_loss, take_profit)))
        self.min += 1
        self.state = self._construct_state()

        portfolio_return, updated_weights = self.calculate_portfolio_return(stop_loss, take_profit)
        self.actions_memory[-1][:, 0] = updated_weights  # Update weights in action memory
        self.portfolio_return_memory.append(portfolio_return)
        self.asset_memory.append(self.portfolio_value)

        self.reward = self.portfolio_value
        return self.state, self.reward, self.terminal, {}

    def calculate_portfolio_return(self, stop_loss, take_profit):
        """Calculate returns for the portfolio, including stop-loss and take-profit handling."""
        portfolio_value = self.cash
        updated_weights = np.zeros_like(self.share_holdings)
        returns = []

        for i, holding in enumerate(self.share_holdings):
            low = self.data_map[self.timestamps[self.min]]['low'].values[i]
            high = self.data_map[self.timestamps[self.min]]['high'].values[i]
            close_price = self.data_map[self.timestamps[self.min]]['close'].values[i]
            open_price = self.data_map[self.timestamps[self.min]]['open'].values[i]
            last_close = self.data_map[self.timestamps[self.min - 1]]['close'].values[i]

            stop_loss_price = last_close * (1 - stop_loss[i])
            take_profit_price = last_close * (1 + take_profit[i])

            # Handle stop-loss and take-profit for long and short positions
            if low <= stop_loss_price and holding > 0:  # Long stop-loss
                current_return = (stop_loss_price - open_price) * holding
                transaction_cost = holding * self.transaction_cost_amount
                current_return -= transaction_cost
                portfolio_value += stop_loss_price * holding
                self.share_holdings[i] = 0
            elif low <= stop_loss_price and holding < 0:  # Short stop-loss
                current_return = (stop_loss_price - open_price) * holding
                transaction_cost = abs(holding) * self.transaction_cost_amount
                current_return -= transaction_cost
                portfolio_value += stop_loss_price * abs(holding)
                self.share_holdings[i] = 0
            elif high >= take_profit_price and holding > 0:  # Long take-profit
                current_return = (take_profit_price - open_price) * holding
                transaction_cost = holding * self.transaction_cost_amount
                current_return -= transaction_cost
                portfolio_value += take_profit_price * holding
                self.share_holdings[i] = 0
            elif high >= take_profit_price and holding < 0:  # Short take-profit
                current_return = (take_profit_price - open_price) * holding
                transaction_cost = abs(holding) * self.transaction_cost_amount
                current_return -= transaction_cost
                portfolio_value += take_profit_price * abs(holding)
                self.share_holdings[i] = 0
            else:  # Regular price change
                current_return = (close_price - open_price) * holding

            # Append return
            returns.append(current_return)

        # Calculate portfolio return
        portfolio_return = sum(returns) / self.portfolio_value
        self.portfolio_value = portfolio_value

        # Update portfolio weights based on new holdings
        for i, holding in enumerate(self.share_holdings):
            updated_weights[i] = (holding * self.data_map[self.timestamps[self.min]]['close'].values[i]) / self.portfolio_value if self.portfolio_value > 0 else 0

        return portfolio_return, updated_weights

    def softmax_normalization(self, actions):
        """Normalize actions to valid weights."""
        actions_exp = np.exp(actions - np.max(actions))
        return actions_exp / np.sum(actions_exp)

    def reset(self):
        """Reset the environment to its initial state."""
        self.min = 0
        self.state = self._construct_state()
        self.portfolio_value = self.initial_amount
        self.cash = self.initial_amount
        self.share_holdings = np.zeros(self.stock_dim)
        self.asset_memory = [self.initial_amount]
        self.portfolio_return_memory = [0]
        self.actions_memory = [[[0]] * self.stock_dim]
        self.date_memory = [self.timestamps[0]]
        return self.state

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
        df_actions.columns = self.data_map[self.timestamps[self.min]]['tic'].values
        df_actions.index = df_date.timestamp
        return df_actions

    def render(self, mode='human'):
        """Render the current environment state."""
        return self.state

    def _seed(self, seed=None):
        """Set random seed for reproducibility."""
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def get_sb_env(self):
        """Get the stable-baselines environment."""
        e = DummyVecEnv([lambda: self])
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









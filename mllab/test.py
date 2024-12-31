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
                 risk_volume,
                 initial_amount,
                 transaction_cost_amount,
                 reward_scaling,
                 tech_indicator_list,
                 features_list,
                 turbulence_threshold=None,
                 lookback=5):
        """
        Initialize the environment with the given parameters.

        Parameters:
        - df: DataFrame containing market data.
        - stock_dim: Number of stocks in the portfolio.
        - hmax: Maximum shares per transaction.
        - initial_amount: Starting portfolio cash value.
        - transaction_cost_amount: Cost per share transaction.
        - reward_scaling: Scaling factor for rewards.
        - state_space: Dimensions of the state space.
        - action_space: Dimensions of the action space.
        - tech_indicator_list: List of technical indicators to include in the state.
        - turbulence_threshold: Optional threshold for turbulence, unused in this version.
        - lookback: Number of historical time steps for constructing the state.
        """
        self.min = 0  # Current time index
        self.lookback = lookback  # Number of previous steps for state construction
        self.df = df  # Market data
        self.stock_dim = stock_dim  # Number of stocks
        self.hmax = hmax  # Max shares per transaction
        self.risk_volume = risk_volume # уровень риска по каждому пакету акций в %
        self.initial_amount = initial_amount  # Starting portfolio cash value
        self.transaction_cost_amount = transaction_cost_amount  # Cost per share transaction
        self.reward_scaling = reward_scaling  # Scaling factor for reward
        self.state_space = (len(features_list) + len(tech_indicator_list)) * lookback  # Dimensions of state space
        self.action_space = 0 #spaces.Box(low=-1, high=1, shape=(stock_dim, 3))  # Long/Short/StopLoss/TakeProfit

        self.tech_indicator_list = tech_indicator_list  # List of technical indicators
        self.features_list = features_list  # List of features

        # Precompute timestamps and data mapping
        self.timestamps = df['timestamp'].sort_values().unique()
        self.data_map = {ts: df[df['timestamp'] == ts] for ts in self.timestamps}
        self.state = self._construct_state()
        self.terminal = False
        self.portfolio_value = self.initial_amount  # Portfolio value
        self.cash = self.initial_amount  # Cash is now
        self.share_holdings = np.zeros(self.stock_dim)  # Share holdings

        # Memory for tracking and logging
        self.asset_memory = [
            {
                'cash': self.initial_amount,
                'portfolio_value': self.initial_amount,
                'holdings': np.zeros(self.stock_dim).tolist()  # Convert array to list for compatibility
            }
        ]
        self.portfolio_return_memory = [0]
        self.actions_memory = [[[0]] * self.stock_dim]
        self.date_memory = [self.timestamps[0]]

    def _construct_state(self):
        """
        Construct the current state with historical data.
        """
        start_index = max(0, self.min - self.lookback + 1)
        historical_data = self.df.iloc[start_index:self.min + 1]

        # Collect technical indicators
        state_data = []

        for feature in self.features_list:
            feature_values = historical_data[feature].values[-self.lookback:] if len(
                historical_data) >= self.lookback else np.zeros(self.lookback)
            state_data.append(feature_values)

        for tech in self.tech_indicator_list:
            tech_values = historical_data[tech].values[-self.lookback:] if len(
                historical_data) >= self.lookback else np.zeros(self.lookback)
            state_data.append(tech_values)

        return np.concatenate(state_data)

    def get_transaction_cost(self, amount, current_price):
        """
        Calculate the transaction cost for a deal.

        Parameters:
        - amount: The number of shares to buy or sell.
        - current_price: The current price of the stock.

        Returns:
        - transaction_cost: The calculated transaction cost.
        """
        value_deal = abs(amount) * current_price
        base_cost = value_deal * self.transaction_cost_amount['base']
        minimal_cost = self.transaction_cost_amount['minimal']
        maximal_cost = self.transaction_cost_amount['maximal'] * value_deal

        # Apply cost boundaries
        transaction_cost = max(minimal_cost, min(base_cost, maximal_cost))

        return transaction_cost

    def _sell_stock(self, stock_index, amount, current_price):
        """
        Sell stock and handle cash and holdings, accounting for long and short positions.

        Parameters:
        - stock_index: Index of the stock to sell.
        - amount: Amount of stock to sell (positive for selling long, negative for short).
        - current_price: Current price of the stock.
        """
        if amount == 0:
            return

        # Calculate transaction cost
        transaction_cost = self.get_transaction_cost(amount, current_price)

        # Update portfolio and cash
        sell_value = amount * current_price
        self.cash += sell_value - transaction_cost
        self.portfolio_value -= transaction_cost
        self.share_holdings[stock_index] -= amount

    def _sell_all_stocks(self):
        """
        Sell all long and short positions.
        """
        for i, holding in enumerate(self.share_holdings):
            current_price = self.data_map[self.timestamps[self.min]]['close'].values[i]
            self._sell_stock(i, holding, current_price)

    def step(self, actions):
        """
        Execute one step in the environment.

        Parameters:
        - actions: Array of actions (weights, stop_loss, take_profit) for each stock.

        Returns:
        - state: Updated state after the step.
        - reward: Reward for the step.
        - terminal: Boolean indicating if the episode is finished.
        - info: Additional information (currently empty).
        """
        self.terminal = self.min >= len(self.timestamps) - 1
        last_minute_of_day = self.min < len(self.timestamps) - 1 and self.timestamps[self.min].date() != \
                             self.timestamps[self.min + 1].date()

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
        weight_diff = np.array(new_weights) - np.array(self.actions_memory[-1][0])

        for i, diff in enumerate(weight_diff):
            current_price = self.data_map[self.timestamps[self.min]]['close'].values[i]
            self._sell_stock(i, int(diff * self.portfolio_value / current_price), current_price)


        self.min += 1
        self.state = self._construct_state()

        portfolio_return, updated_weights = self.calculate_portfolio_return(stop_loss, take_profit)
        self.actions_memory.append(
            np.vstack((updated_weights, stop_loss, take_profit)))  # Update weights in action memory
        self.portfolio_return_memory.append(portfolio_return)
        self.asset_memory.append(
            {'cash': self.cash, 'portfolio_value': self.portfolio_value, 'holdings': self.share_holdings.copy()})

        self.reward = self.portfolio_value * self.reward_scaling
        return self.state, self.reward, self.terminal, {}

    def calculate_portfolio_return(self, stop_loss, take_profit):
        """
        Calculate returns for the portfolio, including stop-loss and take-profit handling.
        """
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
                current_return = (stop_loss_price - last_close) * holding
                transaction_cost = self.get_transaction_cost(holding, stop_loss_price)
                current_return -= transaction_cost
                self.cash += stop_loss_price * holding - transaction_cost
                self.share_holdings[i] = 0
            elif high >= stop_loss_price and holding < 0:  # Short stop-loss
                current_return = (stop_loss_price - last_close) * holding
                transaction_cost = self.get_transaction_cost(holding, stop_loss_price)
                current_return -= transaction_cost
                self.cash += stop_loss_price * holding - transaction_cost
                self.share_holdings[i] = 0
            elif high >= take_profit_price and holding > 0:  # Long take-profit
                current_return = (take_profit_price - last_close) * holding
                transaction_cost = self.get_transaction_cost(holding, take_profit_price)
                current_return -= transaction_cost
                self.cash += take_profit_price * holding - - transaction_cost
                self.share_holdings[i] = 0
            elif low <= take_profit_price and holding < 0:  # Short take-profit
                current_return = (take_profit_price - last_close) * holding
                transaction_cost = self.get_transaction_cost(holding, take_profit_price)
                current_return -= transaction_cost
                self.cash += take_profit_price * holding - transaction_cost
                self.share_holdings[i] = 0
            else:  # Regular price change
                current_return = (close_price - last_close) * holding

            # Append return
            returns.append(current_return)

        # Calculate portfolio return
        portfolio_return = sum(returns) / self.portfolio_value
        self.portfolio_value += sum(returns)

        # Update portfolio weights based on new holdings
        for i, holding in enumerate(self.share_holdings):
            updated_weights[i] = (holding * self.data_map[self.timestamps[self.min]]['close'].values[
                i]) / self.portfolio_value if self.portfolio_value > 0 else 0

        return portfolio_return, updated_weights

    def softmax_normalization(self, actions):
        """
        Normalize actions to valid weights where the sum of absolute weights equals 1,
        and each weight does not exceed self.risk_volume.

        Parameters:
        - actions: Array of raw action weights.

        Returns:
        - normalized_weights: Array of weights balanced and capped by self.risk_volume.
        """
        abs_sum = np.sum(np.abs(actions))

        # Handle the edge case where all actions are zero
        if abs_sum == 0:
            return np.zeros_like(actions)

        # Normalize actions to make the sum of absolute values equal 1
        normalized_actions = actions / abs_sum

        # Cap each normalized action to self.risk_volume
        capped_actions = np.clip(normalized_actions, -self.risk_volume, self.risk_volume)

        # Re-normalize to ensure the sum of absolute weights equals 1 after capping
        abs_capped_sum = np.sum(np.abs(capped_actions))
        if abs_capped_sum == 0:
            return np.zeros_like(actions)  # Handle the case where all weights are capped to zero
        final_weights = capped_actions / abs_capped_sum

        return final_weights

    def reset(self):
        """
        Reset the environment to its initial state.
        """
        self.min = 0
        self.state = self._construct_state()
        self.portfolio_value = self.initial_amount  # Reset portfolio
        self.cash = self.initial_amount  # Reset cash
        self.share_holdings = np.zeros(self.stock_dim)  # Reset share holdings
        self.asset_memory = [
            {
                'cash': self.initial_amount,
                'portfolio_value': self.initial_amount,
                'holdings': np.zeros(self.stock_dim).tolist()  # Convert array to list for compatibility
            }
        ]
        self.portfolio_return_memory = [0]
        self.actions_memory = [[0] * self.stock_dim]
        self.date_memory = [self.timestamps[0]]
        return self.state


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
    'risk_volume': 0.2
    "initial_amount": 1000000,
    "transaction_cost_amount": {'base':0.0035, 'minimal':0.35, 'maximal':0.01},
    "stock_dim": stock_dimension,
    "tech_indicator_list": [],
    'features_list': ['prediction_list', 'trade_list'],
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
    #print(state)
    #print("\nReward after second step:", reward)
    print("\ncash status:", e_train.cash)
    print("\nvalue status:", e_train.portfolio_value)
    print("\nHoldings info:", e_train.share_holdings)
    print("\nreturne rate:", e_train.portfolio_return_memory[-1])









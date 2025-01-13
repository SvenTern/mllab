"""
Implements general backtest statistics.
"""

from mllab.backtest_statistics.backtests import CampbellBacktesting, test_prediction_game, test_label_game, show_heatmap, d3_map, test_sltp_run
from mllab.backtest_statistics.statistics import (timing_of_flattening_and_flips, average_holding_period,
                                                     bets_concentration, all_bets_concentration,
                                                     drawdown_and_time_under_water, sharpe_ratio,
                                                     information_ratio, probabilistic_sharpe_ratio,
                                                     deflated_sharpe_ratio, minimum_track_record_length)

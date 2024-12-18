"""
Labeling techniques used in financial machine learning.
"""

from mllab.labeling.labeling import (add_vertical_barrier, apply_pt_sl_on_t1, barrier_touched, drop_labels,
                                        get_bins, get_events)
from mllab.labeling.trend_scanning import trend_scanning_labels
from mllab.labeling.tail_sets import TailSetLabels
from mllab.labeling.fixed_time_horizon import fixed_time_horizon
from mllab.labeling.matrix_flags import MatrixFlagLabels
from mllab.labeling.excess_over_median import excess_over_median
from mllab.labeling.raw_return import raw_return
from mllab.labeling.return_vs_benchmark import return_over_benchmark
from mllab.labeling.excess_over_mean import excess_over_mean
from mllab.labeling.bull_bear import (pagan_sossounov, lunde_timmermann)

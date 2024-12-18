"""
Various codependence measures: mutual info, distance correlations, variation of information.
"""

from mllab.codependence.correlation import (angular_distance, absolute_angular_distance, squared_angular_distance,
                                               distance_correlation, kullback_leibler_distance, norm_distance)
from mllab.codependence.information import (get_mutual_info, get_optimal_number_of_bins, variation_of_information_score)
from mllab.codependence.codependence_matrix import (get_dependence_matrix, get_distance_matrix)
from mllab.codependence.gnpr_distance import (spearmans_rho, gpr_distance, gnpr_distance)
from mllab.codependence.optimal_transport import (optimal_transport_dependence)

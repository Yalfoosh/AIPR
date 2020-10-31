# Copyright 2020 Yalfoosh
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np

GOLDEN_SECTION_VERBOSITY_DICT = {
    "none": 0,
    "points only": 1,
    "full": 2,
}

GOLDEN_SECTION_K_CONSTANT = (np.sqrt(5) - 1) / 2

NELDER_MEAD_SIMPLEX_VERBOSITY_DICT = {
    "none": 0,
    "point only": 1,
    "full": 2,
}

HOOKE_JEEVES_VERBOSITY_DICT = {
    "none": 0,
    "points only": 1,
    "full": 2,
}

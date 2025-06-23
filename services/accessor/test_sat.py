import torch
import numpy as np
from .sat import prepare_inputs, validate_accessor_output
from hypothesis import given, strategies as st

@st.composite
def random_input(draw):
    n = draw(st.integers(min_value=1, max_value=10))
    time = draw(st.lists(st.floats(min_value=0, max_value=365*10), min_size=n, max_size=n))
    lat = draw(st.lists(st.floats(min_value=18.0, max_value=31.0), min_size=n, max_size=n))
    lon = draw(st.lists(st.floats(min_value=-98.0, max_value=-81.0), min_size=n, max_size=n))
    sss = draw(st.lists(st.floats(min_value=30.0, max_value=40.0), min_size=n, max_size=n))
    sst = draw(st.lists(st.floats(min_value=270.0, max_value=310.0), min_size=n, max_size=n))
    ssh = draw(st.lists(st.floats(min_value=-1.0, max_value=1.0), min_size=n, max_size=n))
    input_params = {
        "timecos": True, "timesin": True, "latcos": True, "latsin": True,
        "loncos": True, "lonsin": True, "sat": True, "sst": True, "sss": True, "ssh": True
    }
    return time, lat, lon, sss, sst, ssh, input_params, n

@given(random_input())
def test_prepare_inputs_shape_and_nanfree(args):
    time, lat, lon, sss, sst, ssh, input_params, n = args
    tensor = prepare_inputs(time, lat, lon, sss, sst, ssh, input_params)
    # 9 features as in the model
    validate_accessor_output(tensor, expected_shape=(n, 9)) 
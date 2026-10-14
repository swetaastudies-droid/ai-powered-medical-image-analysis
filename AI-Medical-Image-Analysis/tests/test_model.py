from src.model import build_model

def test_model_output_shape():
    model = build_model()
    assert model.output_shape == (None, 1)

from src.networks.registry import ALL_TEMPLATES
from src.networks.factory import ModelFactory
from src.networks.estimator import ParameterCalculator

def test_all_templates_exact_math():
    for t in ALL_TEMPLATES:
        m = ModelFactory.create(t)
        actual = sum(p.numel() for p in m.parameters())
        calc = ParameterCalculator.calculate(t)
        assert calc["total_parameters"] == actual, f"Mismatch for {t['id']}: actual={actual}, calc={calc['total_parameters']}"

def test_imports():
    import fastf1
    import lightgbm
    import pandas

    assert fastf1.__version__ and lightgbm.__version__ and pandas.__version__

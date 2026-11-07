import numpy as np
import pytest

from cellsense.validation.datasets import AgeingSeries, fit_fade_shape, forecast_check


def test_fade_shape_fit_on_synthetic_curve():
    c = np.arange(0, 600, 5.0)
    cap = 2.0 * (1 - 0.004 * np.sqrt(c) - 0.0003 * c)
    f = fit_fade_shape(AgeingSeries("synthetic", c, cap))
    assert f["r2"] > 0.999
    assert f["a"] == pytest.approx(0.004, rel=0.02)


def test_forecast_check_returns_error():
    c = np.arange(0, 600, 5.0)
    cap = 2.0 * (1 - 0.002 * np.sqrt(c) - 0.0004 * c)
    fc = forecast_check(AgeingSeries("synthetic", c, cap))
    assert fc["error_cycles"] is not None and abs(fc["error_cycles"]) < 60


def test_every_dashboard_page_builds():
    from cellsense.dashboard import app as A
    from cellsense.dashboard.session import LiveSession

    sess = LiveSession(mode="demo", speed=1, warmup_s=600)
    sess.stop()
    try:
        assert A.create_app(sess).layout is not None
        A.overview(sess)
        A.cells_static(sess, 47)
        A.cells_dynamic(sess, 47)
        A.forecast_page(sess)
        A.lab_static(sess)
        A.lab_dynamic(sess)
        A.forecast_lab_page(sess)
        sess.inject("resistance_growth", 29, 0.35)
        for _ in range(60):
            sess._tick()
        sess.snap = sess.engine.snapshot()
        A.lab_dynamic(sess)
        A.lab_dynamic(sess, compact=True)
        A.forecast_lab_page(sess)
    finally:
        sess.stop()


def test_extract_nasa_handles_nested_zips_and_ignores_unsafe_names(tmp_path):
    import io
    import zipfile

    from cellsense.validation.datasets import extract_nasa

    inner = io.BytesIO()
    with zipfile.ZipFile(inner, "w") as z:
        z.writestr("BatteryAgingARC-FY08Q4/B0005.mat", b"mat-bytes")
        z.writestr("README.txt", b"ignored")
    outer = tmp_path / "5. Battery Data Set.zip"
    with zipfile.ZipFile(outer, "w") as z:
        z.writestr("1. BatteryAgingARC-FY08Q4.zip", inner.getvalue())
        z.writestr("../evil.mat", b"x")
    out = extract_nasa(outer, tmp_path / "out")
    assert sorted(p.name for p in out.iterdir()) == ["B0005.mat", "evil.mat"]
    assert (out / "B0005.mat").read_bytes() == b"mat-bytes"
    assert not (tmp_path / "evil.mat").exists()  # flattened into out/, never written outside it


def test_lab_buttons_only_write_inside_their_own_page():
    """Dash fires a callback when its inputs appear on screen if any of its outputs was already there, which
    would inject a fault every time the Forecast or Fault lab page opened."""
    from cellsense.dashboard import app as A
    from cellsense.dashboard.session import LiveSession

    sess = LiveSession(mode="demo", speed=1, warmup_s=60)
    sess.stop()
    keys = [k for k, v in A.create_app(sess).callback_map.items() if any(i["id"] == "btn-inject" for i in v["inputs"])]
    assert len(keys) == 1
    assert [o.split("@")[0] for o in keys[0].strip(".").split("...")] == ["lab-status.children"]  # the key lists the outputs


@pytest.mark.parametrize("argv", [
    ["dashboard", "--speed", "0"],
    ["dashboard", "--port", "99999"],
    ["dashboard", "--replay", "no_such_run.npz"],
    ["train", "--packs", "0"],
    ["benchmark", "--workers", "-1"],
    ["validate", "--oxford", "no_such_file.mat"],
])
def test_cli_rejects_bad_arguments_with_a_usage_error(argv, capsys):
    from cellsense.__main__ import main

    with pytest.raises(SystemExit) as e:
        main(argv)
    assert e.value.code == 2  # argparse usage error, not a traceback
    assert "error: argument" in capsys.readouterr().err

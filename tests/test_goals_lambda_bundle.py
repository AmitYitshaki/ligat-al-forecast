import pytest

from ligat.evaluate import goals_lambda_bundle as bundle
from ligat.evaluate import lambda_layer
from ligat.evaluate.goals_experiment import load_goal_experiment
from ligat.evaluate.goals_runner import goal_provenance, load_goal_inputs


@pytest.mark.real_data
def test_published_bundle_matches_all_signed_folds_and_rejects_modified_files(
    real_tables, tmp_path, monkeypatch
):
    monkeypatch.setattr(
        lambda_layer, "run_goal_units", lambda *a, **kw: pytest.fail("Export attempted to fit")
    )
    spec = load_goal_experiment()
    data = load_goal_inputs(spec, tables=real_tables)
    export_data = bundle.layer_data(data)
    assert goal_provenance(data) == goal_provenance(export_data)
    manifest = bundle.publish(spec, data, tmp_path)
    assert bundle.load_bundle_summary(spec, data, tmp_path) == manifest
    assert list(manifest["models"]["f5"]["nll"]) == ["all", "deep", "last5"]
    for key, record in manifest["models"].items():
        assert len(record["unit_ids"]) == 15
        assert record["rows"] == 7442
        assert record["games"] == 3721
        assert record["seasons"] == 15
        assert record["deep_seasons"] == (13 if key == "glm_base" else 14)
    report = bundle._signed(spec.output / "full" / "ladder_report.json")
    configs = bundle.signed_configurations(report)
    with pytest.raises(ValueError, match="signature"):
        lambda_layer.load_lambda_layer(
            configs["maher"], export_data, tmp_path / "f5", evaluation_root=spec.output
        )
    table = tmp_path / "f5" / "lambda.parquet"
    original = table.read_bytes()
    table.write_bytes(bytes([original[0] ^ 1]) + original[1:])
    with pytest.raises(ValueError, match="parquet signature"):
        lambda_layer.load_lambda_layer(
            configs["f5"], export_data, tmp_path / "f5", evaluation_root=spec.output
        )
    table.write_bytes(original)
    folds = tmp_path / "glm_base" / "folds.parquet"
    raw = folds.read_bytes()
    folds.write_bytes(bytes([raw[0] ^ 1]) + raw[1:])
    with pytest.raises(ValueError, match="Bundle signature mismatch"):
        bundle.load_bundle_summary(spec, data, tmp_path)


@pytest.mark.real_data
def test_missing_last_native_model_prevents_all_publication(real_tables, tmp_path, monkeypatch):
    spec = load_goal_experiment()
    data = load_goal_inputs(spec, tables=real_tables)
    real_status = bundle.goal_units_status

    def missing_maher(units, *args, **kwargs):
        if units[0].backend == "maher":
            return [{"status": "missing"}]
        return real_status(units, *args, **kwargs)

    monkeypatch.setattr(bundle, "goal_units_status", missing_maher)
    with pytest.raises(ValueError, match="maher"):
        bundle.publish(spec, data, tmp_path)
    assert list(tmp_path.iterdir()) == []

import csv

import pytest

from app.services.variance import VarianceAnalyzer, VarianceDataError


def _write_csv(path, rows):
    fieldnames = ["line_item", "department", "budget", "actual"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_over_budget_expense_is_unfavorable(tmp_path):
    csv_path = tmp_path / "data.csv"
    _write_csv(csv_path, [{"line_item": "Marketing Spend", "department": "Marketing",
                            "budget": 100, "actual": 150}])

    results = VarianceAnalyzer(str(csv_path)).run()

    assert len(results) == 1
    r = results[0]
    assert r.direction == "over"
    assert r.variance == 50
    assert r.variance_pct == 50.0
    assert "unfavorable" in r.commentary


def test_revenue_under_budget_is_unfavorable(tmp_path):
    csv_path = tmp_path / "data.csv"
    _write_csv(csv_path, [{"line_item": "Revenue", "department": "Sales",
                            "budget": 1000, "actual": 900}])

    results = VarianceAnalyzer(str(csv_path)).run()

    r = results[0]
    assert r.direction == "under"
    assert "unfavorable" in r.commentary


def test_revenue_over_budget_is_favorable(tmp_path):
    csv_path = tmp_path / "data.csv"
    _write_csv(csv_path, [{"line_item": "Revenue", "department": "Sales",
                            "budget": 1000, "actual": 1100}])

    results = VarianceAnalyzer(str(csv_path)).run()

    r = results[0]
    assert "favorable" in r.commentary
    assert "unfavorable" not in r.commentary


def test_zero_variance_has_no_investigation_needed(tmp_path):
    csv_path = tmp_path / "data.csv"
    _write_csv(csv_path, [{"line_item": "Rent", "department": "Operations",
                            "budget": 500, "actual": 500}])

    results = VarianceAnalyzer(str(csv_path)).run()

    assert "on budget" in results[0].commentary


def test_missing_required_column_raises(tmp_path):
    csv_path = tmp_path / "bad.csv"
    with open(csv_path, "w") as f:
        f.write("line_item,department,budget\nRevenue,Sales,1000\n")  # no "actual"

    with pytest.raises(VarianceDataError):
        VarianceAnalyzer(str(csv_path)).run()


def test_missing_file_raises(tmp_path):
    with pytest.raises(VarianceDataError):
        VarianceAnalyzer(str(tmp_path / "does_not_exist.csv")).run()
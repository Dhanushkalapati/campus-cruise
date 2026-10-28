import pytest

from conftest import write_data
from src.data import load_dataset

FIXED_CSV = "Bus size,Fixed price\n15,40\n35,60\n50,75\n"


def test_real_dataset_matches_supplied_files(real_dataset):
    d = real_dataset
    assert d.report.ok, d.report.errors
    assert len(d.schools) == 60
    assert d.schools["attendees"].sum() == 1244
    assert d.schools["school_id"].is_unique
    assert d.rates["fixed"] == {15: 40.0, 35: 60.0, 50: 75.0}
    assert d.rates["distance"] == {15: 2.25, 35: 3.25, 50: 4.0}
    assert d.rates["time"] == {15: 135.0, 35: 195.0, 50: 240.0}
    assert d.available_models == ["fixed", "distance", "time"]
    assert all(not v for v in d.report.capacity_errors.values())


def test_empty_school_file(tmp_path):
    d = load_dataset(write_data(tmp_path, "", FIXED_CSV))
    assert not d.report.ok
    assert d.schools.empty


def test_header_only_school_file(tmp_path):
    d = load_dataset(write_data(tmp_path, "outcode,latitude,longitude,attendants\n", FIXED_CSV))
    assert not d.report.ok


def test_missing_columns(tmp_path):
    d = load_dataset(write_data(tmp_path, "outcode,lat\nA,51\n", FIXED_CSV))
    assert any("missing columns" in e for e in d.report.errors)


def test_malformed_and_missing_coordinate_rows_are_excluded(tmp_path):
    text = (
        "outcode,latitude,longitude,attendants\n"
        "A,51.5,-3.2,10\n"
        "B,,-3.2,10\n"          # missing latitude
        "C,51.5,abc,10\n"       # malformed longitude
        "D,95,-3.2,10\n"        # out of range
        "E,51.5,-3.2,-4\n"      # negative attendance
        "F,51.5,-3.2,2.5\n"     # fractional attendance
    )
    d = load_dataset(write_data(tmp_path, text, FIXED_CSV))
    assert d.report.ok
    assert d.schools["school_id"].tolist() == ["A"]
    assert len(d.report.excluded) == 5


def test_duplicate_ids_block_optimisation(tmp_path):
    text = "outcode,latitude,longitude,attendants\nA,51.5,-3.2,10\nA,51.6,-3.1,5\n"
    d = load_dataset(write_data(tmp_path, text, FIXED_CSV))
    assert any("Duplicate" in e for e in d.report.errors)


def test_zero_attendance_is_warned(tmp_path):
    text = "outcode,latitude,longitude,attendants\nA,51.5,-3.2,0\nB,51.6,-3.1,5\n"
    d = load_dataset(write_data(tmp_path, text, FIXED_CSV))
    assert d.report.ok
    assert any("zero attendees" in w for w in d.report.warnings)


def test_invalid_and_negative_costs_excluded(tmp_path):
    text = "outcode,latitude,longitude,attendants\nA,51.5,-3.2,10\n"
    fixed = "Bus size,Fixed price\n15,-40\n35,abc\n0,60\n50,75\n"
    d = load_dataset(write_data(tmp_path, text, fixed))
    assert d.rates["fixed"] == {50: 75.0}
    assert sum("row excluded" in w for w in d.report.warnings) == 3


def test_all_costs_invalid_blocks(tmp_path):
    text = "outcode,latitude,longitude,attendants\nA,51.5,-3.2,10\n"
    d = load_dataset(write_data(tmp_path, text, "Bus size,Fixed price\n15,-1\n"))
    assert not d.report.ok
    assert d.available_models == []


def test_school_larger_than_every_bus(tmp_path):
    text = "outcode,latitude,longitude,attendants\nA,51.5,-3.2,51\n"
    d = load_dataset(write_data(tmp_path, text, FIXED_CSV))
    assert not d.report.model_ok("fixed")
    assert "cannot be split" in d.report.capacity_errors["fixed"][0]


def test_attendance_override_does_not_change_original(real_dataset):
    before = real_dataset.schools["attendees"].copy()
    edited = real_dataset.with_attendance({"SA6": 5})
    assert int(edited.schools.loc[edited.schools.school_id == "SA6", "attendees"].iloc[0]) == 5
    assert real_dataset.schools["attendees"].equals(before)


@pytest.mark.parametrize("bad", [-1, 2.5])
def test_attendance_override_rejects_bad_values(real_dataset, bad):
    with pytest.raises(ValueError):
        real_dataset.with_attendance({"SA6": bad})


def test_attendance_override_rechecks_capacity(real_dataset):
    edited = real_dataset.with_attendance({"SA6": 60})
    assert not edited.report.model_ok("fixed")
    assert real_dataset.report.model_ok("fixed")

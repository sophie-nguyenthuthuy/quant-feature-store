"""Survivorship-bias-free universe: delisted symbols must appear in
membership queries dated BEFORE their delisting, even if 'currently'
they don't exist."""

from __future__ import annotations

from datetime import datetime

from qfs import Universe


def test_delisted_symbol_visible_before_delisting(tmp_path):
    u = Universe(tmp_path)
    # Lehman: in the universe until bankruptcy Sept 15, 2008.
    u.add("LEH", datetime(2000, 1, 1), datetime(2008, 9, 15),
          knowledge_time=datetime(2000, 1, 1))
    # A survivor that never left.
    u.add("AAPL", datetime(2000, 1, 1), None,
          knowledge_time=datetime(2000, 1, 1))

    members_2007 = u.members_as_of(datetime(2007, 6, 1))
    members_2010 = u.members_as_of(datetime(2010, 1, 1))

    assert "LEH" in members_2007 and "AAPL" in members_2007
    assert "LEH" not in members_2010 and "AAPL" in members_2010


def test_membership_respects_knowledge_time(tmp_path):
    """If we only RECORDED the delisting on 2008-10-01 (a month after the
    fact), then a backtest run before that date should still see LEH as
    a member — it didn't know about the delisting yet."""
    u = Universe(tmp_path)
    # First recorded with no end date.
    u.add("LEH", datetime(2000, 1, 1), included_to=None,
          knowledge_time=datetime(2000, 1, 1))
    # Then corrected — same start date, with end date this time.
    u.add("LEH", datetime(2000, 1, 1), included_to=datetime(2008, 9, 15),
          knowledge_time=datetime(2008, 10, 1))

    # As of Sept 20, 2008: we still 'think' LEH is a member (the
    # correction won't be entered until Oct 1).
    assert "LEH" in u.members_as_of(datetime(2008, 9, 20))
    # As of Nov 1, 2008: correction is in, LEH is gone.
    assert "LEH" not in u.members_as_of(datetime(2008, 11, 1))


def test_empty_before_first_addition(tmp_path):
    u = Universe(tmp_path)
    u.add("AAPL", datetime(2024, 1, 1), knowledge_time=datetime(2024, 1, 1))
    assert u.members_as_of(datetime(2023, 12, 1)) == []


def test_unknown_membership_not_visible(tmp_path):
    u = Universe(tmp_path)
    u.add("AAPL", datetime(2024, 1, 1), knowledge_time=datetime(2024, 6, 1))
    # Membership starts Jan 1 but we didn't RECORD it until Jun 1.
    # A backtest as-of March would not have known.
    assert u.members_as_of(datetime(2024, 3, 1)) == []
    assert u.members_as_of(datetime(2024, 7, 1)) == ["AAPL"]

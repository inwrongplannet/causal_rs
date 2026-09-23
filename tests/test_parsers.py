from src.data_pipeline.parsers import parse_entities, parse_history, parse_impressions


class TestParseHistory:
    def test_normal_input(self):
        assert parse_history("N123 N456 N789") == ["N123", "N456", "N789"]

    def test_empty_string(self):
        assert parse_history("") == []

    def test_whitespace_only(self):
        assert parse_history("   ") == []

    def test_nan_input(self):
        assert parse_history(float("nan")) == []

    def test_single_item(self):
        assert parse_history("N55528") == ["N55528"]

    def test_extra_spaces(self):
        assert parse_history("  N123   N456  ") == ["N123", "N456"]


class TestParseImpressions:
    def test_normal_clicked(self):
        result = parse_impressions("N123-1 N456-0 N789-1")
        assert result == [("N123", 1), ("N456", 0), ("N789", 1)]

    def test_empty_string(self):
        assert parse_impressions("") == []

    def test_nan_input(self):
        assert parse_impressions(float("nan")) == []

    def test_missing_dash(self):
        assert parse_impressions("N123 N456") == []

    def test_mixed_format(self):
        result = parse_impressions("N123-1 N456-0")
        assert result == [("N123", 1), ("N456", 0)]

    def test_item_id_after_dash(self):
        result = parse_impressions("-1")
        assert result == []


class TestParseEntities:
    def test_valid_json_list(self):
        blob = '[{"WikidataId": "Q42", "Label": "EntityA"}, {"WikidataId": "Q99", "Label": "EntityB"}]'
        assert parse_entities(blob) == ["Q42", "Q99"]

    def test_valid_json_single(self):
        blob = '[{"WikidataId": "Q42"}]'
        assert parse_entities(blob) == ["Q42"]

    def test_empty_array(self):
        assert parse_entities("[]") == []

    def test_nan_blob(self):
        assert parse_entities(float("nan")) == []

    def test_missing_wikidata_id_falls_back_to_label(self):
        blob = '[{"Label": "EntityA"}]'
        assert parse_entities(blob) == ["EntityA"]

    def test_not_a_list(self):
        assert parse_entities('{"key": "value"}') == []

    def test_malformed_json(self):
        assert parse_entities("{bad json}") == []

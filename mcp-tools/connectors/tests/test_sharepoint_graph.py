"""Tests for the Graph client error mapping + response parsing.

Exercise the parts that are deterministic without a live Graph: the
status-code-to-exception mapping and the search-response decoder.
The HTTP plumbing itself (request method, URL construction) is
covered indirectly via the connector tests with httpx MockTransport.
"""
from __future__ import annotations

import pathlib
import sys

import pytest

_CONNECTORS_DIR = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_CONNECTORS_DIR))

from _base import ReauthenticationRequiredError  # noqa: E402
from sharepoint.graph import (  # noqa: E402
    GraphError,
    GraphNotFoundError,
    GraphRateLimitedError,
    GraphServerError,
    SharePointGraphClient,
)


# ─── error mapping ────────────────────────────────────────────────────


class TestErrorMapping:
    def test_200_does_not_raise(self):
        SharePointGraphClient._raise_for_status(200, "", "sharepoint")

    def test_201_does_not_raise(self):
        SharePointGraphClient._raise_for_status(201, "", "sharepoint")

    def test_204_does_not_raise(self):
        SharePointGraphClient._raise_for_status(204, "", "sharepoint")

    def test_401_raises_reauth(self):
        with pytest.raises(ReauthenticationRequiredError) as excinfo:
            SharePointGraphClient._raise_for_status(401, "Unauthorized", "sharepoint")
        assert excinfo.value.server_name == "sharepoint"
        assert "401" in excinfo.value.reason

    def test_403_raises_reauth(self):
        with pytest.raises(ReauthenticationRequiredError):
            SharePointGraphClient._raise_for_status(403, "Forbidden", "sharepoint")

    def test_404_raises_not_found(self):
        with pytest.raises(GraphNotFoundError) as excinfo:
            SharePointGraphClient._raise_for_status(404, "Not Found", "sharepoint")
        assert excinfo.value.status == 404

    def test_429_raises_rate_limited(self):
        with pytest.raises(GraphRateLimitedError):
            SharePointGraphClient._raise_for_status(429, "Too many", "sharepoint")

    def test_500_raises_server_error(self):
        with pytest.raises(GraphServerError):
            SharePointGraphClient._raise_for_status(500, "Boom", "sharepoint")

    def test_503_raises_server_error(self):
        with pytest.raises(GraphServerError):
            SharePointGraphClient._raise_for_status(503, "Down", "sharepoint")

    def test_unknown_status_raises_generic(self):
        with pytest.raises(GraphError) as excinfo:
            SharePointGraphClient._raise_for_status(418, "I'm a teapot", "sharepoint")
        assert excinfo.value.status == 418

    def test_body_truncated_in_error_message(self):
        long_body = "x" * 5000
        with pytest.raises(GraphError) as excinfo:
            SharePointGraphClient._raise_for_status(500, long_body, "sharepoint")
        # Body should be truncated to ~200 chars in the message
        assert len(str(excinfo.value)) < 500


# ─── search response parser ───────────────────────────────────────────


class TestParseSearchResponse:
    def test_empty_payload(self):
        assert SharePointGraphClient._parse_search_response({}) == []

    def test_empty_value_array(self):
        assert SharePointGraphClient._parse_search_response({"value": []}) == []

    def test_empty_hits_container(self):
        payload = {
            "value": [{
                "hitsContainers": [{"hits": []}]
            }]
        }
        assert SharePointGraphClient._parse_search_response(payload) == []

    def test_basic_hit(self):
        payload = {
            "value": [{
                "hitsContainers": [{
                    "hits": [{
                        "summary": "Quarterly status",
                        "rank": 1,
                        "resource": {
                            "@odata.type": "#microsoft.graph.driveItem",
                            "id": "01ABC",
                            "name": "Q1 Status.docx",
                            "webUrl": "https://example.sharepoint.com/...",
                            "lastModifiedDateTime": "2026-04-15T10:00:00Z",
                            "parentReference": {"driveId": "DRIVE1"},
                        },
                    }]
                }]
            }]
        }
        hits = SharePointGraphClient._parse_search_response(payload)
        assert len(hits) == 1
        h = hits[0]
        assert h.item_id == "01ABC"
        assert h.drive_id == "DRIVE1"
        assert h.name == "Q1 Status.docx"
        assert h.summary == "Quarterly status"
        assert h.score == 1.0
        assert h.last_modified == "2026-04-15T10:00:00Z"

    def test_hit_without_id_skipped(self):
        payload = {
            "value": [{
                "hitsContainers": [{
                    "hits": [{
                        "resource": {"name": "no id"},
                    }]
                }]
            }]
        }
        assert SharePointGraphClient._parse_search_response(payload) == []

    def test_multiple_hits(self):
        payload = {
            "value": [{
                "hitsContainers": [{
                    "hits": [
                        {"resource": {"id": "1", "name": "a", "parentReference": {"driveId": "D"}}, "rank": 1},
                        {"resource": {"id": "2", "name": "b", "parentReference": {"driveId": "D"}}, "rank": 2},
                        {"resource": {"id": "3", "name": "c", "parentReference": {"driveId": "D"}}, "rank": 3},
                    ]
                }]
            }]
        }
        hits = SharePointGraphClient._parse_search_response(payload)
        assert len(hits) == 3
        assert [h.item_id for h in hits] == ["1", "2", "3"]

    def test_invalid_score_falls_back_to_zero(self):
        payload = {
            "value": [{
                "hitsContainers": [{
                    "hits": [{
                        "resource": {"id": "X", "name": "x", "parentReference": {"driveId": "D"}},
                        "rank": "not-a-number",
                    }]
                }]
            }]
        }
        hits = SharePointGraphClient._parse_search_response(payload)
        assert hits[0].score == 0.0

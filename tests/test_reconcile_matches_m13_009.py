from app.reconcile_matches import ReconcileMatchesRequest, suggest_reconcile_matches


def test_suggest_reconcile_matches_m13_009():
    req = ReconcileMatchesRequest(
        unmatchedBank=[
            {
                "id": "bl-1",
                "description": "Transferencia proveedor Acme spa",
                "amount": "-10000",
                "postedDate": "2026-08-05",
            }
        ],
        unmatchedJournal=[
            {
                "id": "je-1",
                "description": "Pago proveedor Acme",
                "netAmount": "-10000",
                "date": "2026-08-06",
            }
        ],
        dateToleranceDays=5,
        amountTolerance=20,
        minConfidence=0.3,
    )
    matches = suggest_reconcile_matches(req)
    assert len(matches) == 1
    assert matches[0]["matchType"] == "ai"
    assert matches[0]["bankLineId"] == "bl-1"
    assert matches[0]["journalEntryId"] == "je-1"
    assert matches[0]["confidence"] >= 0.3

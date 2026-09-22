from rag_ingestion.normalize import (
    TIMELINE_ROLES,
    address_key,
    infer_date_role,
    is_project_token,
    iso_date,
    normalize_lot,
    project_tokens_in_text,
    site_id_for,
)


def test_normalize_lot_strips_spaces_and_words():
    assert normalize_lot("lot 2 363 352") == "2363352"
    assert normalize_lot("2363352") == "2363352"
    assert normalize_lot("4405") is None
    assert normalize_lot("") is None


def test_site_id_prefers_lot_over_address():
    assert site_id_for(lot="2363352", address="619, route 341") == "lot:2363352"
    addr_id = site_id_for(lot=None, address="619, route 341, L'Épiphanie")
    assert addr_id is not None and addr_id.startswith("addr:")
    same = site_id_for(lot=None, address="619 Route 341, L’Épiphanie")
    assert address_key("619, route 341, L'Épiphanie") == address_key(
        "619 Route 341, L’Épiphanie"
    )
    assert same == addr_id


def test_project_tokens_skip_years():
    assert project_tokens_in_text("contamination 4405 en 2019") == ["4405"]
    assert is_project_token("2019") is False
    assert is_project_token("2259") is True


def test_iso_date_and_roles():
    assert iso_date("2023-01-03") == "2023-01-03"
    assert iso_date("3 janvier 2023") is None
    assert infer_date_role("report", "3 janvier 2023") == "report"
    assert infer_date_role("contract", "2 mars 2021") == "contract"
    assert infer_date_role("analysis", "17 août 2019") == "lab_analysis"
    assert infer_date_role("lab_certificate", "2025-09-03") == "lab_certificate"
    assert infer_date_role("sampling", "25-08-2025") == "sampling"
    assert infer_date_role("visit", "28 avril 2021") == "site_visit"
    assert infer_date_role("fieldwork", "17 août 2019") == "fieldwork"
    assert infer_date_role(None, "Date du contrat") == "contract"
    assert infer_date_role(None, "Date de l'analyse") == "lab_analysis"
    assert infer_date_role(None, "Date d'émission du certificat") == "lab_certificate"
    assert infer_date_role(None, "demande d'analyse") == "lab_request"
    assert infer_date_role(None, "demande d'accès à l'information") == "information_request"
    assert infer_date_role(None, "Réponse du MELCCFP reçue le") == "information_response"
    assert infer_date_role(None, "Visite du site") == "site_visit"
    assert infer_date_role(None, "Campagne de forages") == "fieldwork"
    assert infer_date_role(None, "Date du rapport") == "report"
    assert infer_date_role("other", "Entrevues avec le propriétaire") == "unknown"
    assert infer_date_role(None, "quelque part") == "unknown"
    assert TIMELINE_ROLES == {
        "contract",
        "site_visit",
        "fieldwork",
        "sampling",
        "information_request",
        "information_response",
        "lab_request",
        "lab_receipt",
        "lab_analysis",
        "lab_certificate",
        "report",
    }

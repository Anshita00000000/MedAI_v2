"""FHIR R4 Bundle builder for clinical analysis results."""

import logging
import uuid
from datetime import datetime
from typing import List, Optional

logger = logging.getLogger(__name__)


def _uuid() -> str:
    return str(uuid.uuid4())


def _now_iso() -> str:
    return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")


class FHIRBuilder:
    """Build FHIR R4 compliant document bundles."""

    def build_bundle(
        self,
        soap,
        ddx,
        risk,
        treatment,
        patient_id: Optional[str] = None,
    ) -> dict:
        """Build a FHIR R4 Bundle containing the full clinical analysis."""
        patient_id = patient_id or _uuid()
        bundle_id = _uuid()
        timestamp = _now_iso()

        entries = []

        # Composition (SOAP note)
        composition = self._build_composition(soap, patient_id, timestamp)
        entries.append({"fullUrl": f"urn:uuid:{composition['id']}", "resource": composition})

        # Condition resources (differentials)
        if ddx and ddx.differentials:
            for diff in ddx.differentials:
                condition = self._build_condition(diff, patient_id)
                entries.append({"fullUrl": f"urn:uuid:{condition['id']}", "resource": condition})

        # Observation resources (symptoms from soap subjective as proxy)
        if soap and soap.icd_codes:
            for code in soap.icd_codes:
                obs = self._build_observation(code, patient_id, timestamp)
                entries.append({"fullUrl": f"urn:uuid:{obs['id']}", "resource": obs})

        # MedicationRequest resources
        if treatment and treatment.medications:
            for med in treatment.medications:
                med_req = self._build_medication_request(med, patient_id, timestamp)
                entries.append(
                    {"fullUrl": f"urn:uuid:{med_req['id']}", "resource": med_req}
                )

        # RiskAssessment
        if risk:
            risk_assessment = self._build_risk_assessment(risk, patient_id, timestamp)
            entries.append(
                {"fullUrl": f"urn:uuid:{risk_assessment['id']}", "resource": risk_assessment}
            )

        bundle = {
            "resourceType": "Bundle",
            "id": bundle_id,
            "type": "document",
            "timestamp": timestamp,
            "entry": entries,
        }

        self._validate(bundle)
        return bundle

    # ------------------------------------------------------------------ #
    # Resource builders
    # ------------------------------------------------------------------ #

    def _build_composition(self, soap, patient_id: str, timestamp: str) -> dict:
        comp_id = _uuid()
        sections = []
        for section_name, attr in [
            ("Subjective", "subjective"),
            ("Objective", "objective"),
            ("Assessment", "assessment"),
            ("Plan", "plan"),
        ]:
            text = getattr(soap, attr, "") or ""
            sections.append(
                {
                    "title": section_name,
                    "text": {
                        "status": "generated",
                        "div": f'<div xmlns="http://www.w3.org/1999/xhtml">{text}</div>',
                    },
                }
            )

        return {
            "resourceType": "Composition",
            "id": comp_id,
            "status": "final",
            "type": {
                "coding": [
                    {
                        "system": "http://loinc.org",
                        "code": "11488-4",
                        "display": "Consultation note",
                    }
                ]
            },
            "subject": {"reference": f"Patient/{patient_id}"},
            "date": timestamp,
            "title": "MedAI Clinical SOAP Note",
            "section": sections,
        }

    def _build_condition(self, differential, patient_id: str) -> dict:
        cond_id = _uuid()
        coding = [{"display": differential.disease_name}]
        if differential.icd10_code:
            coding.insert(0, {
                "system": "http://hl7.org/fhir/sid/icd-10",
                "code": differential.icd10_code,
                "display": differential.disease_name,
            })
        if differential.umls_code:
            coding.append({
                "system": "http://terminology.hl7.org/CodeSystem/umls",
                "code": differential.umls_code,
            })

        return {
            "resourceType": "Condition",
            "id": cond_id,
            "clinicalStatus": {
                "coding": [
                    {
                        "system": "http://terminology.hl7.org/CodeSystem/condition-clinical",
                        "code": "active",
                    }
                ]
            },
            "verificationStatus": {
                "coding": [
                    {
                        "system": "http://terminology.hl7.org/CodeSystem/condition-ver-status",
                        "code": "differential",
                    }
                ]
            },
            "code": {"coding": coding, "text": differential.disease_name},
            "subject": {"reference": f"Patient/{patient_id}"},
            "note": [{"text": differential.reasoning}],
        }

    def _build_observation(self, icd_code: str, patient_id: str, timestamp: str) -> dict:
        obs_id = _uuid()
        return {
            "resourceType": "Observation",
            "id": obs_id,
            "status": "final",
            "code": {
                "coding": [
                    {
                        "system": "http://hl7.org/fhir/sid/icd-10",
                        "code": icd_code,
                    }
                ]
            },
            "subject": {"reference": f"Patient/{patient_id}"},
            "effectiveDateTime": timestamp,
        }

    def _build_medication_request(self, medication, patient_id: str, timestamp: str) -> dict:
        med_id = _uuid()
        notes = []
        if medication.notes:
            notes.append({"text": medication.notes})

        dose_instruction = {}
        if medication.dose or medication.frequency:
            dose_instruction["text"] = f"{medication.dose} {medication.frequency}".strip()
        if medication.duration:
            dose_instruction["timing"] = {
                "repeat": {"boundsPeriod": {"start": timestamp}},
                "code": {"text": medication.duration},
            }

        return {
            "resourceType": "MedicationRequest",
            "id": med_id,
            "status": "active",
            "intent": "order",
            "medicationCodeableConcept": {"text": medication.name},
            "subject": {"reference": f"Patient/{patient_id}"},
            "authoredOn": timestamp,
            "dosageInstruction": [dose_instruction] if dose_instruction else [],
            "note": notes,
        }

    def _build_risk_assessment(self, risk, patient_id: str, timestamp: str) -> dict:
        ra_id = _uuid()
        urgency_map = {
            "CRITICAL": "0.95",
            "HIGH": "0.75",
            "MEDIUM": "0.50",
            "LOW": "0.20",
        }
        prob = urgency_map.get(risk.urgency, "0.50")

        return {
            "resourceType": "RiskAssessment",
            "id": ra_id,
            "status": "final",
            "subject": {"reference": f"Patient/{patient_id}"},
            "occurrenceDateTime": timestamp,
            "prediction": [
                {
                    "outcome": {"text": risk.urgency},
                    "probabilityDecimal": float(prob),
                    "rationale": risk.reasoning,
                }
            ],
            "note": [
                {"text": risk.recommended_action},
                *[{"text": f"Red flag: {f}"} for f in risk.red_flags],
            ],
        }

    @staticmethod
    def _validate(bundle: dict) -> None:
        assert bundle.get("resourceType") == "Bundle", "Missing resourceType: Bundle"
        assert bundle.get("type") == "document", "Missing type: document"
        assert "timestamp" in bundle, "Missing timestamp"
        assert "entry" in bundle, "Missing entry list"

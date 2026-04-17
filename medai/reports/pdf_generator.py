"""Clinical PDF report generation using WeasyPrint + Jinja2."""

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

TEMPLATES_DIR = Path(__file__).parent / "templates"

try:
    from jinja2 import Environment, FileSystemLoader, select_autoescape
    _jinja_env = Environment(
        loader=FileSystemLoader(str(TEMPLATES_DIR)),
        autoescape=select_autoescape(["html"]),
    )

    def _conf_class(value: float) -> str:
        if value >= 0.75:
            return "high"
        if value >= 0.45:
            return "medium"
        return "low"

    _jinja_env.filters["confidence_class"] = _conf_class
    _JINJA_AVAILABLE = True
except ImportError:
    _JINJA_AVAILABLE = False
    logger.warning("Jinja2 not available; PDF generation disabled.")

try:
    from weasyprint import HTML as WeasyprintHTML
    _WEASYPRINT_AVAILABLE = True
except ImportError:
    _WEASYPRINT_AVAILABLE = False
    logger.warning("WeasyPrint not available; PDF generation disabled.")


class PDFGenerator:
    """Generate clinical PDF reports."""

    def __init__(self):
        self.available = _JINJA_AVAILABLE and _WEASYPRINT_AVAILABLE
        if not self.available:
            logger.warning(
                "PDFGenerator unavailable: requires jinja2 and weasyprint."
            )

    def generate_doctor_report(
        self,
        soap,
        ddx,
        risk,
        treatment,
    ) -> bytes:
        """Generate doctor-facing clinical report as PDF bytes."""
        html = self._render_doctor_template(soap, ddx, risk, treatment)
        return self._html_to_pdf(html)

    def generate_patient_summary(
        self,
        soap,
        treatment,
        urgent_symptoms: Optional[list] = None,
    ) -> bytes:
        """Generate patient-friendly summary as PDF bytes."""
        html = self._render_patient_template(soap, treatment, urgent_symptoms)
        return self._html_to_pdf(html)

    def _render_doctor_template(self, soap, ddx, risk, treatment) -> str:
        if not _JINJA_AVAILABLE:
            raise RuntimeError("Jinja2 not installed.")
        template = _jinja_env.get_template("doctor_report.html")
        return template.render(
            soap=soap,
            ddx=ddx,
            risk=risk,
            treatment=treatment,
            generated_at=datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
        )

    def _render_patient_template(
        self, soap, treatment, urgent_symptoms: Optional[list]
    ) -> str:
        if not _JINJA_AVAILABLE:
            raise RuntimeError("Jinja2 not installed.")
        template = _jinja_env.get_template("patient_summary.html")

        # Build plain-language summary from SOAP subjective
        summary_text = (
            soap.subjective.replace("# Subjective:", "").strip()
            if soap.subjective
            else "Your consultation details are summarised below."
        )

        return template.render(
            summary_text=summary_text,
            medications=getattr(treatment, "medications", []),
            patient_education=getattr(treatment, "patient_education", []),
            follow_up=getattr(treatment, "follow_up", ""),
            referrals=getattr(treatment, "referrals", []),
            urgent_symptoms=urgent_symptoms or [],
            generated_at=datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
        )

    def _html_to_pdf(self, html: str) -> bytes:
        if not _WEASYPRINT_AVAILABLE:
            raise RuntimeError("WeasyPrint not installed.")
        return WeasyprintHTML(string=html).write_pdf()

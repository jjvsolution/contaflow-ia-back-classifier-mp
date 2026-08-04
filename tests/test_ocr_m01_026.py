"""M01-026: OCR servicio — parseo CL + upload PDF retorna RUT/folio/montos/fecha."""

from __future__ import annotations

import io
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.main import app, verify_internal
from app.ocr_parse import parse_invoice_fields
from app.ocr_rut import compute_rut_dv, parse_chile_rut
from app.ocr_service import run_ocr


SAMPLE_INVOICE_TEXT = """
FACTURA ELECTRONICA
RUT Emisor: 76.543.210-3
FOLIO: 45821
Fecha de Emision: 15/03/2026
Razon Social: Demo SpA
Neto: 100.000
IVA 19%: 19.000
Total: 119.000
"""


def _make_text_pdf(text: str) -> bytes:
    """PDF mínimo con capa de texto (Helvetica) para pruebas sin ReportLab."""
    # Escape paréntesis para operadores PDF.
    safe = (
        text.replace("\\", "\\\\")
        .replace("(", "\\(")
        .replace(")", "\\)")
        .replace("\r", " ")
        .replace("\n", " ")[:400]
    )
    stream = f"BT /F1 10 Tf 40 750 Td ({safe}) Tj ET"
    stream_bytes = stream.encode("latin-1", errors="replace")
    objects = [
        b"1 0 obj<< /Type /Catalog /Pages 2 0 R >>endobj\n",
        b"2 0 obj<< /Type /Pages /Kids [3 0 R] /Count 1 >>endobj\n",
        (
            b"3 0 obj<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            b"/Contents 4 0 R /Resources<< /Font<< /F1 5 0 R >> >> >>endobj\n"
        ),
        (
            f"4 0 obj<< /Length {len(stream_bytes)} >>stream\n".encode("ascii")
            + stream_bytes
            + b"\nendstream\nendobj\n"
        ),
        b"5 0 obj<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>endobj\n",
    ]
    out = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for obj in objects:
        offsets.append(len(out))
        out.extend(obj)
    xref_pos = len(out)
    out.extend(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
    out.extend(b"0000000000 65535 f \n")
    for off in offsets[1:]:
        out.extend(f"{off:010d} 00000 n \n".encode("ascii"))
    out.extend(
        f"trailer<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
        f"startxref\n{xref_pos}\n%%EOF\n".encode("ascii")
    )
    return bytes(out)


class OcrRutTests(unittest.TestCase):
    def test_dv_conocido(self):
        self.assertEqual(compute_rut_dv("76543210"), "3")
        parsed = parse_chile_rut("76.543.210-3")
        assert parsed is not None
        self.assertEqual(parsed["rutNormalized"], "765432103")
        self.assertEqual(parsed["rutDisplay"], "76.543.210-3")

    def test_rut_invalido(self):
        self.assertIsNone(parse_chile_rut("76.543.210-1"))


class OcrParseTests(unittest.TestCase):
    def test_extrae_campos_factura_tipica(self):
        fields = parse_invoice_fields(SAMPLE_INVOICE_TEXT)
        self.assertEqual(fields["rut"], "76.543.210-3")
        self.assertEqual(fields["rutNormalized"], "765432103")
        self.assertEqual(fields["folio"], "45821")
        self.assertEqual(fields["issueDate"], "2026-03-15")
        self.assertEqual(fields["amountNet"], "100000")
        self.assertEqual(fields["amountVat"], "19000")
        self.assertEqual(fields["totalAmount"], "119000")
        self.assertEqual(fields["currency"], "CLP")

    def test_factura_sii_fecha_espanol_e_iva_con_puntos(self):
        text = """
JJV SOLUTIONS SPA
Giro: SERVICIOS
TIPO DE
VENTA: DEL GIRO
SEÑOR(ES): SERVICIOS Y ASESORIAS RUBRIKA S.A
R.U.T.: 76.190.474- 4
TIPO DE
COMPRA: DEL GIRO
R.U.T.:78.107.147- 1
FACTURA ELECTRONICA
Nº12
Fecha Emision: 02 de Diciembre del 2025
MONTO NETO $ 1.415.000
I.V.A. 19% $ 268.850
IMPUESTO ADICIONAL $ 0
TOTAL $ 1.683.850
"""
        fields = parse_invoice_fields(text)
        self.assertEqual(fields["folio"], "12")
        self.assertEqual(fields["issueDate"], "2025-12-02")
        self.assertEqual(fields["amountNet"], "1415000")
        self.assertEqual(fields["amountVat"], "268850")
        self.assertEqual(fields["amountExempt"], "0")
        self.assertEqual(fields["totalAmount"], "1683850")
        self.assertEqual(fields["rutIssuer"], "78.107.147-1")
        self.assertEqual(fields["rutReceiver"], "76.190.474-4")
        self.assertEqual(fields["issuerName"], "JJV SOLUTIONS SPA")
        self.assertEqual(
            fields["receiverName"], "SERVICIOS Y ASESORIAS RUBRIKA S.A"
        )
        self.assertEqual(fields["siiDocumentType"], "33")
        self.assertEqual(fields["siiOperationTypeSale"], "DEL GIRO")

    def test_boleta_honorarios_guion_unicode_y_n_espacio(self):
        text = """
JOSEPH ALEXANDER VENEGAS ZURITA
BOLETA DE HONORARIOS
ELECTRONICA
N ° 11
RUT: 21.092.011−0
Fecha: 13 de Mayo de 2024
Señor(es): SERVICIOS Y ASESORIAS RUBRIKA S.A Rut: 76.190.474− 4
Total Honorarios: $: 1.477.913
13.75 % Impto. Retenido: 203.213
Total: 1.274.700
Res. Ex. N° 83 de 30/08/2004
El contribuyente receptor de esta boleta debe retener el porcentaje definido.
"""
        fields = parse_invoice_fields(text)
        self.assertEqual(fields["folio"], "11")
        self.assertEqual(fields["issueDate"], "2024-05-13")
        self.assertEqual(fields["rutIssuer"], "21.092.011-0")
        self.assertEqual(fields["rutReceiver"], "76.190.474-4")
        self.assertEqual(
            fields["issuerName"], "JOSEPH ALEXANDER VENEGAS ZURITA"
        )
        self.assertEqual(fields["amountGross"], "1477913")
        self.assertEqual(fields["amountRetention"], "203213")
        self.assertEqual(fields["amountRetentionTaxpayer"], "203213")
        self.assertEqual(fields["amountRetentionThirdParty"], "0")
        self.assertEqual(fields["totalAmount"], "1274700")


class OcrServicePdfTests(unittest.TestCase):
    def test_pdf_text_layer_retorna_campos(self):
        pdf = _make_text_pdf(SAMPLE_INVOICE_TEXT)
        result = run_ocr(
            filename="factura.pdf",
            content_type="application/pdf",
            data=pdf,
        )
        self.assertTrue(result["ok"])
        self.assertEqual(result["engine"], "pypdf")
        fields = result["fields"]
        self.assertEqual(fields["rutNormalized"], "765432103")
        self.assertEqual(fields["folio"], "45821")
        self.assertEqual(fields["issueDate"], "2026-03-15")
        self.assertEqual(fields["totalAmount"], "119000")
        self.assertEqual(fields["amountNet"], "100000")


class OcrHttpTests(unittest.TestCase):
    def setUp(self):
        app.dependency_overrides[verify_internal] = lambda: None
        self.client = TestClient(app)

    def tearDown(self):
        app.dependency_overrides.clear()

    def test_upload_pdf_retorna_rut_folio_montos_fecha(self):
        pdf = _make_text_pdf(SAMPLE_INVOICE_TEXT)
        res = self.client.post(
            "/v1/ocr",
            files={"file": ("factura-demo.pdf", pdf, "application/pdf")},
        )
        self.assertEqual(res.status_code, 200, res.text)
        body = res.json()
        self.assertTrue(body["ok"])
        fields = body["fields"]
        self.assertEqual(fields["rut"], "76.543.210-3")
        self.assertEqual(fields["folio"], "45821")
        self.assertEqual(fields["issueDate"], "2026-03-15")
        self.assertEqual(fields["amountNet"], "100000")
        self.assertEqual(fields["amountVat"], "19000")
        self.assertEqual(fields["totalAmount"], "119000")

    def test_archivo_vacio_400(self):
        res = self.client.post(
            "/v1/ocr",
            files={"file": ("vacio.pdf", b"", "application/pdf")},
        )
        self.assertEqual(res.status_code, 400)

    def test_imagen_sin_tesseract_responde_ok_con_warning(self):
        # PNG 1x1 mínimo
        png = (
            b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
            b"\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00"
            b"\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00\x05\x18"
            b"\xd8N\x00\x00\x00\x00IEND\xaeB`\x82"
        )
        with patch("app.ocr_engine._tesseract_available", return_value=False):
            res = self.client.post(
                "/v1/ocr",
                files={"file": ("scan.png", png, "image/png")},
            )
        self.assertEqual(res.status_code, 200, res.text)
        body = res.json()
        self.assertEqual(body["engine"], "none")
        self.assertTrue(any("Tesseract" in w for w in body["warnings"]))


if __name__ == "__main__":
    unittest.main()

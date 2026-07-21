"""Contrato Pydantic alineado a contaflow-ia-back `llm.types.ts` (M01-019)."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

DocumentKind = Literal["purchase", "sale", "fee", "bank_statement_line"]
LlmPurpose = Literal[
    "classify_purchase",
    "classify_sale",
    "classify_fee",
    "classify_bank_line",
    "suggest_journal_entry",
]
LanguageHint = Literal["es", "en"]
ClassifyMode = Literal["suggest", "classify_only"]
NormalBalance = Literal["debit", "credit"]


class _StrictIgnore(BaseModel):
    model_config = ConfigDict(extra="ignore")


class Money(_StrictIgnore):
    amount: str
    currency: str


class PeriodRef(_StrictIgnore):
    companyId: str
    fiscalYear: int
    month: int = Field(ge=1, le=12)
    periodId: str | None = None
    isClosed: bool | None = None


class CompanyContext(_StrictIgnore):
    companyId: str
    giro: str
    country: Literal["CL"] | None = None
    industryTags: list[str] | None = None


class ChartAccountRef(_StrictIgnore):
    name: str
    accountId: str | None = None
    code: str | None = None
    normalBalance: NormalBalance | None = None


class SourcePayload(_StrictIgnore):
    textRaw: str | None = None
    textRedacted: str | None = None
    languageHint: LanguageHint | None = None


class TotalsPayload(_StrictIgnore):
    net: Money | None = None
    tax: Money | None = None
    exempt: Money | None = None
    total: Money | None = None


class BankPayload(_StrictIgnore):
    bankName: str | None = None
    statementLineId: str | None = None
    postedDate: str | None = None
    memo: str | None = None


class StructuredPayload(_StrictIgnore):
    documentNumber: str | None = None
    counterpartyName: str | None = None
    counterpartyTaxIdMasked: str | None = None
    issueDate: str | None = None
    totals: TotalsPayload | None = None
    bank: BankPayload | None = None


class AccountingRules(_StrictIgnore):
    vatTypicalRate: float | None = None
    requireCostCenter: bool | None = None


class AccountingContext(_StrictIgnore):
    chartOfAccountsTop: list[ChartAccountRef] | None = None
    rules: AccountingRules | None = None


class ClassificationOptions(_StrictIgnore):
    mode: ClassifyMode | None = None
    allowExternalLLM: bool | None = None
    maxCandidates: int | None = None
    explain: bool | None = None


class ClassificationInput(_StrictIgnore):
    requestId: str
    kind: DocumentKind
    period: PeriodRef
    company: CompanyContext
    source: SourcePayload
    tenantId: str | None = None
    structured: StructuredPayload | None = None
    accountingContext: AccountingContext | None = None
    options: ClassificationOptions | None = None


class PromptPayload(_StrictIgnore):
    system: str | None = None
    user: str | None = None
    outputSchemaName: str | None = None
    version: str | None = None


class LlmRequest(_StrictIgnore):
    """Body de POST /v1/classify (= LlmRequest en Nest)."""

    requestId: str
    purpose: LlmPurpose
    input: ClassificationInput
    prompt: PromptPayload | None = None

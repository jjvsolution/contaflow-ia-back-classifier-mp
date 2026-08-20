"""Contrato Pydantic para POST /v1/chat (M01-027 / E20-005)."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class _StrictIgnore(BaseModel):
    model_config = ConfigDict(extra="ignore")


class ChatMessage(_StrictIgnore):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=8000)


class ChatCompanyContext(_StrictIgnore):
    companyId: str
    name: str | None = None
    giro: str | None = None
    taxId: str | None = None


class ChatPeriodContext(_StrictIgnore):
    periodId: str
    fiscalYear: int
    month: int = Field(ge=1, le=12)
    status: str | None = None
    isClosed: bool | None = None


class ChatAccountRef(_StrictIgnore):
    accountId: str | None = None
    code: str | None = None
    name: str


class ChatRequest(_StrictIgnore):
    requestId: str | None = None
    tenantId: str
    message: str = Field(min_length=1, max_length=4000)
    history: list[ChatMessage] = Field(default_factory=list, max_length=20)
    company: ChatCompanyContext
    period: ChatPeriodContext | None = None
    chartOfAccounts: list[ChatAccountRef] = Field(default_factory=list, max_length=200)


class ChatSuggestedLine(_StrictIgnore):
    accountCode: str | None = None
    accountName: str | None = None
    debit: str | None = None
    credit: str | None = None
    memo: str | None = None


class ChatSuggestedEntry(_StrictIgnore):
    memo: str | None = None
    lines: list[ChatSuggestedLine] = Field(default_factory=list)


class ChatResponse(_StrictIgnore):
    requestId: str
    reply: str
    citedAccounts: list[ChatAccountRef] = Field(default_factory=list)
    suggestedEntry: ChatSuggestedEntry | None = None
    requiresHumanApproval: bool = True
    registeredJournalEntry: bool = False
    ragStatus: Literal["ok", "degraded", "failed"] = "degraded"
    provider: dict

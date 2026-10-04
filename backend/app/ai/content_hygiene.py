"""Narrow user-facing text hygiene and methodology-claim grounding checks."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from app.ai.grounding import EvidenceCategory, EvidenceRecord
from app.ai.schemas import ToolName
from app.ai.validation import ValidationFinding, ValidationResolution

_CONTROL_CHARACTERS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_GOVERNED_TERMS: dict[str, tuple[str, ...]] = {
    "archetype": ("archetype",),
    "off-ball": ("off-ball", "off ball"),
    "centroid boundary": ("centroid boundary",),
    "net forward carry": ("net forward carry",),
    "pairwise distance": ("pairwise distance", "pairwise-distance"),
    "primary driver": ("primary driver", "primary style driver"),
}
_ROLE_FIT_BAND = re.compile(
    r"\b(?:excellent|good|moderate|poor|strong|weak|low|high)"
    r"(?:\s*/\s*[a-z-]+)?\s+(?:stylistic\s+)?(?:role[ -]?fit|fit)\b|"
    r"\b(?:role[ -]?fit|fit)\s+(?:is\s+|appears\s+|looks\s+|seems\s+)?"
    r"(?:excellent|good|moderate|poor|strong|weak|low|high)\b|"
    r"\b(?:excellent|good|moderate|poor|strong|weak|low|high)[ -]distance"
    r"(?:\s+(?:role[ -]?fit|fit|match))?\b|"
    r"\b(?:excellent|good|moderate|poor|strong|weak|low|high)\s+"
    r"(?:stylistic\s+)?match\b|"
    r"\b(?:most\s+)?(?:moderately|strongly|weakly|poorly)\s+"
    r"(?:close|align(?:s|ed)?|match(?:es|ed)?)\b|"
    r"\bmost\s+(?:closely\s+)?align(?:s|ed)?\b|"
    r"\bstrongest\s+(?:stylistic\s+)?"
    r"(?:alignment|dimensions?|role[ -]?fit|fit)\b|"
    r"\b(?:excellent|good|moderate|poor|strong|weak)\s+"
    r"(?:stylistic\s+)?alignment\b|\bnot\s+extreme\b",
    re.IGNORECASE,
)
_ROLE_FIT_DRIVER_LANGUAGE = re.compile(
    r"^[ \t]{0,3}(?:#{1,6}[ \t]*)?(?:key|main|primary)[ \t]+"
    r"(?:(?:stylistic|style|role[ -]?fit)[ \t]+)?drivers?[ \t]*#*[ \t]*$|"
    r"\b(?:stylistic|style|role[ -]?fit)\s+drivers?\b|"
    r"\b(?:closest|nearest)\s+(?:observed\s+)?dimensions?\b"
    r"[^.!?\n]{0,100}\b(?:drives?|drivers?|causes?|main\s+reason)\b|"
    r"\b(?:drives?|drivers?|causes?|main\s+reason)\b"
    r"[^.!?\n]{0,100}\b(?:closest|nearest)\s+(?:observed\s+)?dimensions?\b",
    re.IGNORECASE | re.MULTILINE,
)
_ROLE_FIT_LIMITATION_EXPANSIONS: dict[str, tuple[str, ...]] = {
    "availability": ("availability", "lineup availability"),
    "injury status": ("injury status", "injury", "injuries"),
    "selection probability": ("selection probability", "selection likelihood"),
}
_ROLE_FIT_NON_PREDICTION_CLAIM = re.compile(
    r"\bdo(?:es)?\s+not\s+predict\b(?P<scope>[^.;!?\n]{0,180})",
    re.IGNORECASE,
)
_ROLE_FIT_PREDICTION_SCOPE_EVIDENCE = re.compile(
    r"\b(?:do(?:es)?\s+not\s+predict|"
    r"(?:is|are)\s+not\s+(?:a\s+)?predictions?\s+of)\b"
    r"(?P<scope>[^.;!?\n]{0,180})",
    re.IGNORECASE,
)
_ROLE_FIT_PREDICTION_SCOPE_CONNECTORS = frozenset(
    {"a", "an", "and", "any", "either", "neither", "nor", "or", "the"}
)
_NORMATIVE_STYLE_LANGUAGE = re.compile(
    r"\b(?:needs?|requires?)\s+improvement\b|"
    r"\bneeds?\s+to\s+improve\b|"
    r"\b(?:must|should)\s+improve\b|"
    r"\bweakness(?:es)?\b|\bdeficienc(?:y|ies)\b|\bbetter\b|\bworse\b",
    re.IGNORECASE,
)
_NORMATIVE_CLAUSE = re.compile(r"[^.;:!?\n]+")
_CONTRAST_BOUNDARY = re.compile(r"\b(?:but|however|yet)\b", re.IGNORECASE)
_DIRECT_NORMATIVE_NEGATION = re.compile(
    r"\b(?:not|no)\s+(?:an?\s+)?$",
    re.IGNORECASE,
)
_NORMATIVE_DISCLAIMER = re.compile(
    r"\b(?:do(?:es)?|did|should|would|must|is|are|was|were)\s+not\s+"
    r"(?:mean|imply|indicate|show|suggest|measure|represent|constitute|"
    r"be\s+interpreted(?:\s+as)?|be\s+treated(?:\s+as)?)\b|"
    r"\b(?:is|are|was|were)\s+not\s+(?:evidence|proof|an?\s+indication)\s+of\b",
    re.IGNORECASE,
)
_IMPLEMENTATION_CLAIM_PATTERNS: dict[str, re.Pattern[str]] = {
    "matching": re.compile(
        r"\b(?:deterministic|fuzzy|substring|normalized|exact)\s+"
        r"(?:name\s+|entity\s+)?match(?:ing|er|es)?\b|"
        r"\b(?:player\s+)?(?:identity|name|entity|resolver|query|search)\b"
        r"[^.!?\n]{0,90}\b(?:deterministic|fuzzy|substring|normalized|exact)\b"
        r"[^.!?\n]{0,50}\b(?:match(?:ing|er|es)?|resolution)\b",
        re.IGNORECASE,
    ),
    "retrieval": re.compile(
        r"\b(?:returns?|returned|retrieves?|retrieved|fetches?|fetched|exposes?|"
        r"exposed|provides?|provided)\b[^.!?\n]{0,90}\braw\s+(?:event\s+)?rows?\b|"
        r"\braw\s+(?:event\s+)?rows?\b[^.!?\n]{0,90}\b(?:returns?|returned|"
        r"retrieves?|retrieved|fetches?|fetched|exposes?|exposed)\b",
        re.IGNORECASE,
    ),
    "api": re.compile(
        r"\b(?:api|endpoint)\b[^.!?\n]{0,100}\b(?:returns?|retrieves?|fetches?|"
        r"exposes?|accepts?|calls?|queries?)\b|"
        r"\b(?:returns?|retrieves?|fetches?|exposes?|accepts?|calls?|queries?)\b"
        r"[^.!?\n]{0,100}\b(?:api|endpoint)\b",
        re.IGNORECASE,
    ),
    "storage": re.compile(
        r"\b(?:database|storage|postgresql|sql|table|parquet)\b[^.!?\n]{0,100}"
        r"\b(?:stores?|persists?|reads?|writes?|queries?|retrieves?|uses?)\b|"
        r"\b(?:stores?|persists?|reads?|writes?|queries?|retrieves?)\b"
        r"[^.!?\n]{0,100}\b(?:database|storage|postgresql|sql|table|parquet)\b",
        re.IGNORECASE,
    ),
    "tool": re.compile(
        r"\b(?:internal\s+)?tools?\b[^.!?\n]{0,100}\b(?:calls?|runs?|returns?|"
        r"retrieves?|fetches?|queries?|uses?)\b|"
        r"\b(?:calls?|runs?|returns?|retrieves?|fetches?|queries?|uses?)\b"
        r"[^.!?\n]{0,100}\b(?:internal\s+)?tools?\b",
        re.IGNORECASE,
    ),
}
_IMPLEMENTATION_DETAIL_TERMS = {
    "api",
    "database",
    "deterministic",
    "endpoint",
    "event",
    "exact",
    "fuzzy",
    "normalized",
    "parquet",
    "postgres",
    "postgresql",
    "raw",
    "row",
    "rows",
    "sql",
    "storage",
    "substring",
    "table",
    "tool",
    "tools",
}
_METHODOLOGY_STRENGTHENING_ASSERTION = re.compile(
    r"\b(?:tuned|optimized|optimised|hyperparameter[ -]?tuned|grid[ -]?searched|"
    r"automatically\s+selected|searched|swept)\b|"
    r"\bselected\s+(?:through|by|using)\s+(?:a\s+)?(?:search|sweep|optimization)\b",
    re.IGNORECASE,
)
_METHODOLOGY_TUNING_PROCESS = re.compile(
    r"\b(?:tun(?:e|ed|ing)|optimi[sz](?:e|ed|ation|ing)|"
    r"hyperparameter[ -]?(?:tuning|search)|grid[ -]?search|validation\s+search|"
    r"parameter\s+search|search(?:ed|ing)?\s+(?:over|across|for)|"
    r"sweep(?:ed|ing)?|automatically\s+selected)\b",
    re.IGNORECASE,
)
_EXPLICIT_EXTERNAL_WORLD_ASSERTION = re.compile(
    r"\b(?:plays?\s+for|manages|coaches|has\s+joined|joined|signed\s+for|"
    r"transferred\s+(?:to|from)|moved\s+(?:to|from)|is\s+staying\s+at)\b|"
    r"\b(?:current|present)\s+(?:club|team|manager|coach|injury\s+status|"
    r"availability|transfer\s+status|squad|lineup|status)\s*(?:is|remains|:)\s*\S|"
    r"\b(?:current|latest|recent)\s+(?:public\s+)?(?:reporting|reports?|news|"
    r"update)\b|"
    r"\btransfer\s+(?:report|reports|reporting|rumou?r|rumou?rs|news|update)\b|"
    r"\bavailable\s+for\s+(?:today|tonight|tomorrow|the\s+(?:next|upcoming))\b|"
    r"\b(?:currently|now)\s+(?:represents|is\s+(?:with|at)|starts?|"
    r"is\s+in\s+the\s+lineup)\b|"
    r"\bis\s+(?:currently|now)\s+(?:with|at)\b|"
    r"\bshould\s+sign\b[^.!?\n]{0,100}\bnow\b",
    re.IGNORECASE,
)
_EXTERNAL_PERSON_STATUS_ASSERTION = re.compile(
    r"\b(?:(?:[A-Z][\w.'’&-]*)(?:\s+[A-Z][\w.'’&-]*){0,3}|"
    r"(?i:he|she|the\s+player|player\s+[a-z][\w.'’&-]*))\s+"
    r"(?:is|remains)\s+(?:currently\s+)?"
    r"(?:injured|available|unavailable|sidelined)\b",
)
_PRESENT_PLAYER_IDENTITY_ASSERTION = re.compile(
    r"\b(?:is|remains)\s+(?:currently\s+)?(?:an?\s+)?"
    r"(?:[A-Z][\w.'’&-]*\s+){0,4}player\b",
)
_CONTEXTUAL_EXTERNAL_WORLD_ASSERTION = re.compile(
    r"\b(?:club|team|manager|coach|squad|lineup|injury\s+status|availability|"
    r"transfer\s+status)\s*(?:is|remains|:)\s*\S|"
    r"\b(?:has|suffers?\s+from)\s+(?:an?\s+)?(?:[\w-]+\s+){0,3}injury\b|"
    r"\b(?:joined|signed|transferred|departed|leaving)\b|"
    r"\b(?:public\s+)?reporting\s+(?:identifies|describes|says|places|links)\b",
    re.IGNORECASE,
)
_RETRIEVAL_GAP_DISCLOSURE = re.compile(
    r"^(?:[-*]\s*)?(?:"
    r"(?:I\s+)?(?:could\s+not|couldn't|was\s+unable\s+to)\s+verify\s+"
    r"(?:current\s+)?(?:transfer\s+reporting|injury\s+status|availability|"
    r"recent\s+news|current\s+club|current\s+manager|information)\s+from\s+"
    r"qualifying(?:\s+fresh)?(?:\s+public)?\s+sources\s+in\s+this\s+search"
    r"|no\s+qualifying(?:\s+fresh)?(?:\s+transfer-reporting)?\s+sources\s+"
    r"were\s+available\s+in\s+this\s+search"
    r")[.!]?$",
    re.IGNORECASE,
)
_GOVERNED_METHODOLOGY_STATE = re.compile(
    r"\b(?:current\s+)?production\s+(?:[\w-]+\s+){0,3}"
    r"(?:model|approach|system)\b|"
    r"\bproduction\s+(?:[\w-]+\s+){0,3}(?:model|approach|system)\s+"
    r"(?:uses?|is|was|select(?:ed|s))\b|"
    r"\b(?:model|feature|training|evaluation)\s+"
    r"(?:architecture|definition|design|methodology|selection|family)\b|"
    r"\b(?:selected\s+)?(?:model|estimator|approach)\s+(?:uses?|was\s+selected)\b|"
    r"\b(?:cross[ -]?fitting|preprocessing|feature construction|training design)\b|"
    r"\b(?:current|next|previous)\s+(?:event|state|possession)\b|"
    r"\bpre-event\s+(?:game\s+)?state\b",
    re.IGNORECASE,
)
_GOVERNED_METHODOLOGY_SUBJECT = re.compile(
    r"\b(?:model|regressor|classifier|estimator|objective|boosting\s+rounds?|"
    r"validation|model\s+selection|out[ -]of[ -]fold|oof|prediction\s+range|"
    r"preprocessing|cross[ -]?fitting|target|horizon|architecture|"
    r"hyperparameters?)\b|"
    r"\b(?:training|evaluation|feature)\s+"
    r"(?:procedure|process|design|methodology|configuration|uses?|construction)\b|"
    r"\b(?:documented|governed|methodology)\s+limitations?\b|"
    r"\bmethodology\s+(?:version|configuration)\b",
    re.IGNORECASE,
)
_EXPLICIT_CURRENT_TIME_SCOPE = re.compile(
    r"\b(?:currently|latest|recent|recently|now|today|tonight|tomorrow|"
    r"this\s+season)\b|"
    r"\bcurrent\s+(?:club|team|manager|coach|injur(?:y|ed)|availability|"
    r"available|unavailable|sidelined|transfer|squad|lineup|status)\b|"
    r"\btransfer\s+(?:report|reports|reporting|rumou?r|rumou?rs|news|update)\b",
    re.IGNORECASE,
)
_HISTORICAL_DATASET_SCOPE = re.compile(
    r"\bFootyScout(?:['’]s)?\s+(?:historical\s+|observed\s+|analytics\s+)?"
    r"(?:dataset|sample|data)\b|"
    r"\b(?:historical|observed|analytics)\s+FootyScout\s+"
    r"(?:dataset|sample|data)\b|"
    r"\b(?:historical|observed|analytics)\s+(?:dataset|sample|data)\b|"
    r"\bdataset-era\b|\bhistorical observation\b",
    re.IGNORECASE,
)
_HISTORICAL_OBSERVATION = re.compile(
    r"\b(?:was|were|is|are)\s+(?:historically\s+)?(?:recorded|observed|listed)\b|"
    r"\bhistorically\s+(?:recorded|observed|listed)\b|"
    r"\b(?:dataset|sample|data)\s+"
    r"(?:records?|recorded|lists?|listed|contains?|observes?|observed)\b|"
    r"\bhistorical observation\b",
    re.IGNORECASE,
)
_CITATION_GROUP = re.compile(r"\[([^\[\]]+)\]")
_EVIDENCE_ID_TOKEN = re.compile(r"(?=[^\s;\[\]]*evidence)[^\s;\[\]]+", re.IGNORECASE)
_PERFORMANCE_TOOLS = {
    ToolName.GET_PLAYER_DOSSIER,
    ToolName.COMPARE_PLAYERS,
    ToolName.GET_LEADERBOARD,
}
_ROLE_FIT_CALIBRATION_KEYS = {
    "calibrated_thresholds",
    "fit_band",
    "fit_band_thresholds",
    "fit_label",
    "qualitative_fit_band",
    "role_fit_band",
    "role_fit_band_thresholds",
}
_NONFACTUAL_HEADING = re.compile(
    r"(?:historical analysis|role fit(?: analysis)?|summary|limitations|methodology|"
    r"sources|analysis)",
    re.IGNORECASE,
)
_CURRENT_HEADING_VARIANT = re.compile(
    r"(?:current situation|current status|latest update|current update|current context)"
    r"(?:\s*\((?P<descriptor>[^()\n]{1,80})\))?",
    re.IGNORECASE,
)
_EXTERNAL_CURRENT_SECTION_HEADING = re.compile(
    r"(?:current\s+(?:club|team|manager|coach|squad|lineup)|"
    r"injury\s+(?:status|update)|availability(?:\s+update)?|"
    r"transfer\s+(?:status|update|reporting)|(?:latest|recent)\s+news)",
    re.IGNORECASE,
)
_FACTUAL_HEADING_PREDICATE = re.compile(
    r"\b(?:is|are|was|were|has|have|had|plays?|joined?|sign(?:ed|s)?|manages?|"
    r"remains?|injured|available|unavailable|sidelined)\b",
    re.IGNORECASE,
)
_SETEXT_UNDERLINE = re.compile(r"^(?:=+|-+)$")


class GroundingValidationError(ValueError):
    """Safe, structured detail for a rejected user-visible answer claim."""

    def __init__(
        self,
        message: str,
        *,
        error_code: str,
        failed_claim: str | None,
        failed_line: int | None,
        detected_records: tuple[EvidenceRecord, ...] = (),
        detected_evidence_ids: tuple[str, ...] = (),
        detected_evidence_categories: tuple[str, ...] = (),
        rule: str,
        findings: tuple[ValidationFinding, ...] = (),
        resolution: ValidationResolution | None = None,
    ) -> None:
        super().__init__(message)
        self.error_code = error_code
        self.failed_claim = failed_claim
        self.failed_line = failed_line
        self.detected_evidence_ids = tuple(
            dict.fromkeys(
                (
                    *detected_evidence_ids,
                    *(record.evidence_id for record in detected_records),
                )
            )
        )
        self.detected_evidence_categories = tuple(
            dict.fromkeys(
                (
                    *detected_evidence_categories,
                    *(record.evidence_category.value for record in detected_records),
                )
            )
        )
        self.rule = rule
        self.findings = findings
        self.resolution = resolution

    @classmethod
    def from_finding(
        cls,
        finding: ValidationFinding,
        *,
        findings: tuple[ValidationFinding, ...] = (),
        resolution: ValidationResolution | None = None,
    ) -> GroundingValidationError:
        return cls(
            finding.message,
            error_code=finding.error_code,
            failed_claim=finding.claim,
            failed_line=finding.line,
            detected_evidence_ids=finding.evidence_ids,
            detected_evidence_categories=finding.evidence_categories,
            rule=finding.rule,
            findings=findings or (finding,),
            resolution=resolution,
        )


@dataclass(frozen=True)
class _ClaimUnit:
    text: str
    line_number: int
    end_line: int
    current_section: bool


def sanitize_user_facing_text(value: str) -> str:
    """Remove only unsafe control characters, preserving Unicode and punctuation."""
    return _CONTROL_CHARACTERS.sub("", value)


def extract_inline_evidence_ids(value: str) -> tuple[str, ...]:
    """Return individual IDs from single- or multi-evidence citation brackets."""
    evidence_ids: list[str] = []
    for match in _CITATION_GROUP.finditer(value):
        members = tuple(member.strip() for member in match.group(1).split(";"))
        if members and all(
            member and _EVIDENCE_ID_TOKEN.fullmatch(member) for member in members
        ):
            evidence_ids.extend(members)
    return tuple(evidence_ids)


def has_substantive_claims(answer_markdown: str) -> bool:
    """Return whether normalized Markdown contains factual prose beyond headings."""
    return bool(_claim_units(answer_markdown))


def _finding(
    *,
    message: str,
    error_code: str,
    failed_claim: str | None,
    failed_line: int | None,
    rule: str,
    failed_end_line: int | None = None,
    detected_records: tuple[EvidenceRecord, ...] = (),
    detected_evidence_ids: tuple[str, ...] = (),
) -> ValidationFinding:
    return ValidationFinding(
        error_code=error_code,
        rule=rule,
        message=message,
        claim=failed_claim,
        line=failed_line,
        end_line=failed_end_line or failed_line,
        evidence_ids=tuple(
            dict.fromkeys(
                (
                    *detected_evidence_ids,
                    *(record.evidence_id for record in detected_records),
                )
            )
        ),
        evidence_categories=tuple(
            dict.fromkeys(record.evidence_category.value for record in detected_records)
        ),
    )


def _raise_first_finding(findings: tuple[ValidationFinding, ...]) -> None:
    if findings:
        raise GroundingValidationError.from_finding(findings[0], findings=findings)


def validate_grounded_methodology_terms(
    answer_markdown: str,
    cited_records: tuple[EvidenceRecord, ...],
) -> None:
    """Reject governed methodology concepts absent from the cited evidence."""
    _raise_first_finding(
        detect_grounded_methodology_terms(answer_markdown, cited_records)
    )


def detect_grounded_methodology_terms(
    answer_markdown: str,
    cited_records: tuple[EvidenceRecord, ...],
) -> tuple[ValidationFinding, ...]:
    """Detect governed methodology concepts absent from cited evidence."""
    answer_text = answer_markdown.casefold()
    evidence_text = json.dumps(
        [record.result for record in cited_records],
        ensure_ascii=False,
        sort_keys=True,
        default=str,
    ).casefold()
    unsupported = {
        label: variants
        for label, variants in _GOVERNED_TERMS.items()
        if any(variant in answer_text for variant in variants)
        and not any(variant in evidence_text for variant in variants)
    }
    if unsupported:
        failed_claim, failed_line = _first_line_with_terms(
            answer_markdown,
            [variant for variants in unsupported.values() for variant in variants],
        )
        return (
            _finding(
                message=(
                    "Answer introduced methodology terms absent from cited evidence: "
                    f"{sorted(unsupported)}"
                ),
                error_code="unsupported_methodology_term",
                failed_claim=failed_claim,
                failed_line=failed_line,
                detected_records=cited_records,
                rule="methodology_terms_must_appear_in_cited_evidence",
            ),
        )
    return ()


def detect_unsupported_implementation_claims(
    answer_markdown: str,
    cited_records: tuple[EvidenceRecord, ...],
) -> tuple[ValidationFinding, ...]:
    """Detect internal-mechanics assertions absent from claim-local visible evidence."""
    records_by_id = {record.evidence_id: record for record in cited_records}
    findings: list[ValidationFinding] = []
    for claim in _claim_units(answer_markdown):
        categories = _implementation_claim_categories(claim.text)
        if not categories:
            continue
        line_records = tuple(
            records_by_id[evidence_id]
            for evidence_id in extract_inline_evidence_ids(claim.text)
            if evidence_id in records_by_id
        )
        if _implementation_claim_is_supported(claim.text, categories, line_records):
            continue
        findings.append(
            _finding(
                message=(
                    "Answer asserted internal implementation behavior that was not "
                    "explicitly supported by claim-local evidence."
                ),
                error_code="unsupported_implementation_claim",
                failed_claim=claim.text,
                failed_line=claim.line_number,
                failed_end_line=claim.end_line,
                detected_records=line_records,
                rule="implementation_claims_require_explicit_cited_evidence",
            )
        )
    return tuple(findings)


def detect_unsupported_methodology_strengthening(
    answer_markdown: str,
    cited_records: tuple[EvidenceRecord, ...],
) -> tuple[ValidationFinding, ...]:
    """Reject tuning/search claims when evidence supplies only fixed configuration."""
    records_by_id = {record.evidence_id: record for record in cited_records}
    findings: list[ValidationFinding] = []
    for claim in _claim_units(answer_markdown):
        if not _METHODOLOGY_STRENGTHENING_ASSERTION.search(claim.text):
            continue
        line_records = tuple(
            records_by_id[evidence_id]
            for evidence_id in extract_inline_evidence_ids(claim.text)
            if evidence_id in records_by_id
        )
        methodology_records = tuple(
            record
            for record in line_records
            if record.evidence_category is EvidenceCategory.METHODOLOGY
        )
        if not methodology_records:
            continue
        evidence_text = json.dumps(
            [record.result for record in methodology_records],
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        )
        if _METHODOLOGY_TUNING_PROCESS.search(evidence_text):
            continue
        findings.append(
            _finding(
                message=(
                    "Answer strengthened a fixed methodology configuration into an "
                    "unsupported tuning or optimization claim."
                ),
                error_code="unsupported_methodology_strengthening",
                failed_claim=claim.text,
                failed_line=claim.line_number,
                failed_end_line=claim.end_line,
                detected_records=methodology_records,
                rule="methodology_tuning_claims_require_explicit_process_evidence",
            )
        )
    return tuple(findings)


def _implementation_claim_categories(value: str) -> frozenset[str]:
    return frozenset(
        category
        for category, pattern in _IMPLEMENTATION_CLAIM_PATTERNS.items()
        if pattern.search(value)
    )


def _implementation_claim_is_supported(
    claim: str,
    categories: frozenset[str],
    records: tuple[EvidenceRecord, ...],
) -> bool:
    claim_details = _implementation_detail_terms(claim)
    for record in records:
        visible_evidence = json.dumps(
            {"result": record.result, "warnings": record.warnings},
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        )
        evidence_categories = _implementation_claim_categories(visible_evidence)
        if not categories.issubset(evidence_categories):
            continue
        evidence_details = _implementation_detail_terms(visible_evidence)
        if not claim_details or claim_details.issubset(evidence_details):
            return True
    return False


def _implementation_detail_terms(value: str) -> frozenset[str]:
    return frozenset(
        token
        for token in re.findall(r"[a-z]+", value.casefold())
        if token in _IMPLEMENTATION_DETAIL_TERMS
    )


def validate_role_fit_language(
    answer_markdown: str,
    cited_records: tuple[EvidenceRecord, ...],
) -> None:
    """Keep raw Role Fit distance descriptive unless governed evidence says more."""
    _raise_first_finding(detect_role_fit_language(answer_markdown, cited_records))


def detect_role_fit_language(
    answer_markdown: str,
    cited_records: tuple[EvidenceRecord, ...],
) -> tuple[ValidationFinding, ...]:
    """Detect unsupported Role Fit calibration and quality interpretations."""
    role_fit_records = tuple(record for record in cited_records if _is_role_fit(record))
    if not role_fit_records:
        return ()

    findings: list[ValidationFinding] = []
    if not any(
        _has_role_fit_calibration(record.result) for record in role_fit_records
    ):
        band_lines: set[int] = set()
        for band_match in _ROLE_FIT_BAND.finditer(answer_markdown):
            failed_claim, failed_line = _line_at_offset(
                answer_markdown,
                band_match.start(),
            )
            if failed_line in band_lines:
                continue
            band_lines.add(failed_line)
            findings.append(
                _finding(
                    message=(
                        "Answer converted raw Role Fit distance into an uncalibrated "
                        "qualitative band."
                    ),
                    error_code="uncalibrated_role_fit_band",
                    failed_claim=failed_claim,
                    failed_line=failed_line,
                    detected_records=role_fit_records,
                    rule="raw_role_fit_distance_has_no_qualitative_band",
                )
            )

    role_fit_evidence_text = json.dumps(
        [record.result for record in role_fit_records],
        ensure_ascii=False,
        sort_keys=True,
        default=str,
    )
    if not re.search(
        r"\b(?:drivers?|drives|causes?|main reason)\b",
        role_fit_evidence_text.replace("_", " "),
        re.IGNORECASE,
    ):
        driver_lines: set[int] = set()
        for driver_match in _ROLE_FIT_DRIVER_LANGUAGE.finditer(answer_markdown):
            failed_claim, failed_line = _line_at_offset(
                answer_markdown,
                driver_match.start(),
            )
            if failed_line in driver_lines:
                continue
            driver_lines.add(failed_line)
            findings.append(
                _finding(
                    message=(
                        "Answer converted closest observed Role Fit dimensions into "
                        "unsupported causal drivers."
                    ),
                    error_code="unsupported_role_fit_driver_language",
                    failed_claim=failed_claim,
                    failed_line=failed_line,
                    detected_records=role_fit_records,
                    rule="closest_role_fit_dimensions_are_descriptive_not_causal",
                )
            )

    evidence_text = role_fit_evidence_text.casefold().replace("_", " ")
    supported_prediction_scope = _role_fit_prediction_scope_terms(
        role_fit_evidence_text,
        _ROLE_FIT_PREDICTION_SCOPE_EVIDENCE,
    )
    limitation_lines: set[int] = set()
    for claim_match in _ROLE_FIT_NON_PREDICTION_CLAIM.finditer(answer_markdown):
        claim_text = claim_match.group().casefold()
        unsupported_targets = [
            label
            for label, variants in _ROLE_FIT_LIMITATION_EXPANSIONS.items()
            if any(variant in claim_text for variant in variants)
            and not any(variant in evidence_text for variant in variants)
        ]
        claim_prediction_scope = _role_fit_prediction_scope_terms(
            claim_match.group(),
            _ROLE_FIT_NON_PREDICTION_CLAIM,
        )
        if supported_prediction_scope:
            unsupported_scope_terms = sorted(
                claim_prediction_scope - supported_prediction_scope
            )
            if unsupported_scope_terms:
                unsupported_targets.append(
                    "unsupported prediction scope: "
                    + ", ".join(unsupported_scope_terms)
                )
        unsupported_targets = list(dict.fromkeys(unsupported_targets))
        if not unsupported_targets:
            continue
        failed_claim, failed_line = _line_at_offset(
            answer_markdown,
            claim_match.start(),
        )
        if failed_line in limitation_lines:
            continue
        limitation_lines.add(failed_line)
        findings.append(
            _finding(
                message=(
                    "Answer expanded Role Fit limitations beyond cited methodology: "
                    f"{list(unsupported_targets)}"
                ),
                error_code="unsupported_role_fit_limitation_expansion",
                failed_claim=failed_claim,
                failed_line=failed_line,
                detected_records=role_fit_records,
                rule="role_fit_limitations_must_be_explicitly_supported",
            )
        )

    records_by_id = {record.evidence_id: record for record in cited_records}
    for claim in _claim_units(answer_markdown):
        if not _has_unsupported_normative_language(claim.text):
            continue
        line_records = tuple(
            records_by_id[evidence_id]
            for evidence_id in extract_inline_evidence_ids(claim.text)
            if evidence_id in records_by_id
        )
        relevant_records = line_records or role_fit_records
        if any(_is_role_fit(record) for record in relevant_records) and not any(
            record.evidence_category is EvidenceCategory.ANALYTICS
            and record.tool_name in _PERFORMANCE_TOOLS
            for record in line_records
        ):
            findings.append(
                _finding(
                    message=(
                        "Answer treated a Role Fit style difference as a quality "
                        "deficiency."
                    ),
                    error_code="normative_role_fit_inference",
                    failed_claim=claim.text,
                    failed_line=claim.line_number,
                    failed_end_line=claim.end_line,
                    detected_records=line_records,
                    rule="role_fit_style_differences_are_not_quality_judgments",
                )
            )
    return tuple(findings)


def validate_current_claim_authority(
    answer_markdown: str,
    cited_records: tuple[EvidenceRecord, ...],
) -> None:
    """Require web authority on current-world claims while allowing explicit history."""
    _raise_first_finding(detect_current_claim_authority(answer_markdown, cited_records))


def detect_current_claim_authority(
    answer_markdown: str,
    cited_records: tuple[EvidenceRecord, ...],
) -> tuple[ValidationFinding, ...]:
    """Detect current-world claims that lack claim-local web authority."""
    records_by_id = {record.evidence_id: record for record in cited_records}
    findings: list[ValidationFinding] = []
    for claim in _claim_units(answer_markdown):
        line = claim.text
        cited_on_line = tuple(
            records_by_id[evidence_id]
            for evidence_id in extract_inline_evidence_ids(line)
            if evidence_id in records_by_id
        )
        if not _requires_current_world_authority(
            line,
            cited_on_line,
            current_section=claim.current_section,
        ):
            continue
        if not any(
            record.evidence_category is EvidenceCategory.WEB for record in cited_on_line
        ):
            findings.append(
                _finding(
                    message="Current-world claims require a web evidence citation.",
                    error_code="current_world_claim_requires_web",
                    failed_claim=claim.text,
                    failed_line=claim.line_number,
                    failed_end_line=claim.end_line,
                    detected_records=cited_on_line,
                    rule="current_world_claim_requires_at_least_one_web_citation",
                )
            )
    return tuple(findings)


def _requires_current_world_authority(
    text: str,
    cited_records: tuple[EvidenceRecord, ...],
    *,
    current_section: bool,
) -> bool:
    """Identify an asserted mutable external fact, not merely current vocabulary."""
    if _RETRIEVAL_GAP_DISCLOSURE.fullmatch(text.strip()):
        # This is a bounded disclosure about the retrieval attempt, not a claim
        # about the external world (for example, that no rumours exist).
        return False
    if (
        _EXPLICIT_EXTERNAL_WORLD_ASSERTION.search(text)
        or _EXTERNAL_PERSON_STATUS_ASSERTION.search(text)
        or _PRESENT_PLAYER_IDENTITY_ASSERTION.search(text)
    ):
        return True
    if not current_section:
        return False
    if _is_historical_dataset_claim(text):
        return False
    if _is_governed_methodology_state_claim(text, cited_records):
        return False
    return bool(_CONTEXTUAL_EXTERNAL_WORLD_ASSERTION.search(text))


def _is_governed_methodology_state_claim(
    text: str,
    cited_records: tuple[EvidenceRecord, ...],
) -> bool:
    """Separate documented FootyScout model state from external current facts."""
    return bool(
        any(
            record.evidence_category is EvidenceCategory.METHODOLOGY
            for record in cited_records
        )
        and (
            _GOVERNED_METHODOLOGY_STATE.search(text)
            or _GOVERNED_METHODOLOGY_SUBJECT.search(text)
        )
        and not _EXPLICIT_EXTERNAL_WORLD_ASSERTION.search(text)
        and not _EXTERNAL_PERSON_STATUS_ASSERTION.search(text)
        and not _PRESENT_PLAYER_IDENTITY_ASSERTION.search(text)
    )


def _claim_units(answer_markdown: str) -> tuple[_ClaimUnit, ...]:
    """Group soft-wrapped Markdown paragraphs and omit non-factual headings."""
    lines = answer_markdown.splitlines()
    units: list[_ClaimUnit] = []
    buffer: list[str] = []
    start_line = 0
    end_line = 0
    current_section = False
    buffer_section = False

    def flush() -> None:
        nonlocal buffer, start_line, end_line
        if buffer:
            units.append(
                _ClaimUnit(
                    text=" ".join(buffer),
                    line_number=start_line,
                    end_line=end_line,
                    current_section=buffer_section,
                )
            )
            buffer = []
            start_line = 0
            end_line = 0

    index = 0
    while index < len(lines):
        line = lines[index].strip()
        if not line:
            flush()
            index += 1
            continue
        if _is_setext_heading(lines, index) or _is_heading(lines, index):
            flush()
            current_section = _is_external_current_section_heading(
                _heading_text(line)
            )
            if _is_setext_heading(lines, index):
                index += 2
            else:
                index += 1
            continue
        if _SETEXT_UNDERLINE.fullmatch(line):
            index += 1
            continue
        if line.startswith(("- ", "* ", "+ ")) and buffer:
            flush()
        if not buffer:
            start_line = index + 1
            buffer_section = current_section
        buffer.append(line)
        end_line = index + 1
        index += 1
    flush()
    return tuple(units)


def _is_setext_heading(lines: list[str], index: int) -> bool:
    return (
        index + 1 < len(lines)
        and bool(lines[index].strip())
        and bool(_SETEXT_UNDERLINE.fullmatch(lines[index + 1].strip()))
    )


def _is_heading(lines: list[str], index: int) -> bool:
    line = lines[index].strip()
    if re.match(r"^#{1,6}\s+", line):
        return True
    if _NONFACTUAL_HEADING.fullmatch(_heading_text(line)):
        return True
    if _is_current_heading_variant(_heading_text(line)):
        return True
    if bool(
        re.fullmatch(r"(?:\*\*|__)(.+)(?:\*\*|__)", line)
        and not extract_inline_evidence_ids(line)
        and len(_heading_text(line).split()) <= 8
        and not line.rstrip("*_ ").endswith((".", "?", "!"))
    ):
        return True
    return _is_plain_standalone_heading(lines, index)


def _is_plain_standalone_heading(lines: list[str], index: int) -> bool:
    """Recognize only short, isolated legacy titles without factual structure."""
    line = lines[index].strip()
    previous_is_boundary = index == 0 or not lines[index - 1].strip()
    next_is_boundary = index == len(lines) - 1 or not lines[index + 1].strip()
    words = line.split()
    return bool(
        previous_is_boundary
        and next_is_boundary
        and 1 <= len(words) <= 5
        and line[0].isupper()
        and ":" not in line
        and not line.endswith((".", ",", ";", "?", "!"))
        and not extract_inline_evidence_ids(line)
        and not _FACTUAL_HEADING_PREDICATE.search(line)
    )


def _is_current_heading_variant(text: str) -> bool:
    match = _CURRENT_HEADING_VARIANT.fullmatch(text)
    if match is None:
        return False
    descriptor = match.group("descriptor")
    return descriptor is None or not _FACTUAL_HEADING_PREDICATE.search(descriptor)


def _is_external_current_section_heading(text: str) -> bool:
    """Recognize headings that scope prose to mutable external-world facts."""
    return bool(
        _is_current_heading_variant(text)
        or _EXTERNAL_CURRENT_SECTION_HEADING.fullmatch(text)
    )


def _heading_text(line: str) -> str:
    return line.strip().lstrip("#").strip().strip("*_ ").rstrip(":").strip()


def _line_at_offset(answer_markdown: str, offset: int) -> tuple[str, int]:
    line_number = answer_markdown.count("\n", 0, offset) + 1
    lines = answer_markdown.splitlines()
    return (lines[line_number - 1].strip(), line_number)


def _first_line_with_terms(
    answer_markdown: str,
    terms: list[str],
) -> tuple[str | None, int | None]:
    for line_number, line in enumerate(answer_markdown.splitlines(), start=1):
        if any(term in line.casefold() for term in terms):
            return line.strip(), line_number
    return None, None


def _is_role_fit(record: EvidenceRecord) -> bool:
    return (
        record.tool_name in {ToolName.GET_ROLE_FIT, ToolName.GET_ROLE_RECOMMENDATIONS}
        or record.methodology_topic == "role_fit"
    )


def _role_fit_prediction_scope_terms(
    value: str,
    pattern: re.Pattern[str],
) -> frozenset[str]:
    """Return prediction-target terms while ignoring syntax and ledger citations."""
    terms: set[str] = set()
    for match in pattern.finditer(value):
        scope = re.sub(r"\[[^\]]+\]", "", match.group("scope"))
        terms.update(
            token
            for token in re.findall(r"[a-z]+", scope.casefold())
            if token not in _ROLE_FIT_PREDICTION_SCOPE_CONNECTORS
        )
    return frozenset(terms)


def _has_unsupported_normative_language(text: str) -> bool:
    for clause_match in _NORMATIVE_CLAUSE.finditer(text):
        clause = clause_match.group()
        for segment in _CONTRAST_BOUNDARY.split(clause):
            for normative_match in _NORMATIVE_STYLE_LANGUAGE.finditer(segment):
                prefix = segment[: normative_match.start()]
                if _DIRECT_NORMATIVE_NEGATION.search(prefix):
                    continue
                if _NORMATIVE_DISCLAIMER.search(prefix):
                    continue
                return True
    return False


def _is_historical_dataset_claim(text: str) -> bool:
    """Identify scoped historical observations unless current wording overrides them."""
    return bool(
        not _EXPLICIT_CURRENT_TIME_SCOPE.search(text)
        and _HISTORICAL_DATASET_SCOPE.search(text)
        and _HISTORICAL_OBSERVATION.search(text)
    )


def _has_role_fit_calibration(value: object) -> bool:
    if isinstance(value, dict):
        if any(str(key).casefold() in _ROLE_FIT_CALIBRATION_KEYS for key in value):
            return True
        return any(_has_role_fit_calibration(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_has_role_fit_calibration(item) for item in value)
    return False

"""PII (Personally Identifiable Information) detection and redaction."""

import re
from typing import Dict, List, Tuple


# Regex patterns for common PII
EMAIL_PATTERN = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")
PHONE_PATTERN = re.compile(r"\b(?:\+?1[-.]?)?\(?\d{3}\)?[-.]?\d{3}[-.]?\d{4}\b")
SSN_PATTERN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
CREDIT_CARD_PATTERN = re.compile(r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b")
# IP addresses
IP_PATTERN = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")


def detect_pii(text: str) -> Dict[str, List[str]]:
    """
    Detect PII in text using regex patterns.

    Args:
        text: Input text

    Returns:
        Dictionary mapping PII type to list of detected instances
    """
    detected = {}

    # Email addresses
    emails = EMAIL_PATTERN.findall(text)
    if emails:
        detected["email"] = emails

    # Phone numbers
    phones = PHONE_PATTERN.findall(text)
    if phones:
        detected["phone"] = phones

    # SSN
    ssns = SSN_PATTERN.findall(text)
    if ssns:
        detected["ssn"] = ssns

    # Credit cards
    cards = CREDIT_CARD_PATTERN.findall(text)
    if cards:
        detected["credit_card"] = cards

    # IP addresses
    ips = IP_PATTERN.findall(text)
    if ips:
        detected["ip_address"] = ips

    return detected


def redact_pii(text: str, placeholder: str = "[REDACTED]") -> Tuple[str, Dict[str, int]]:
    """
    Redact PII from text.

    Args:
        text: Input text
        placeholder: Replacement text for redacted content

    Returns:
        Tuple of (redacted text, counts of each PII type redacted)
    """
    redacted = text
    counts = {}

    # Redact emails
    email_matches = EMAIL_PATTERN.findall(redacted)
    if email_matches:
        counts["email"] = len(email_matches)
        redacted = EMAIL_PATTERN.sub(f"{placeholder}:EMAIL", redacted)

    # Redact phones
    phone_matches = PHONE_PATTERN.findall(redacted)
    if phone_matches:
        counts["phone"] = len(phone_matches)
        redacted = PHONE_PATTERN.sub(f"{placeholder}:PHONE", redacted)

    # Redact SSN
    ssn_matches = SSN_PATTERN.findall(redacted)
    if ssn_matches:
        counts["ssn"] = len(ssn_matches)
        redacted = SSN_PATTERN.sub(f"{placeholder}:SSN", redacted)

    # Redact credit cards
    card_matches = CREDIT_CARD_PATTERN.findall(redacted)
    if card_matches:
        counts["credit_card"] = len(card_matches)
        redacted = CREDIT_CARD_PATTERN.sub(f"{placeholder}:CARD", redacted)

    # Redact IP addresses (optional - might be too aggressive)
    # ip_matches = IP_PATTERN.findall(redacted)
    # if ip_matches:
    #     counts["ip_address"] = len(ip_matches)
    #     redacted = IP_PATTERN.sub(f"{placeholder}:IP", redacted)

    return redacted, counts


def has_pii(text: str) -> bool:
    """
    Check if text contains any detectable PII.

    Args:
        text: Input text

    Returns:
        True if PII detected
    """
    detected = detect_pii(text)
    return len(detected) > 0

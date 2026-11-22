"""URL feature extraction shared by model training and Flask inference."""
import ipaddress
import math
import re
from collections import Counter
from urllib.parse import urlparse, urlunsplit

SHORTENERS = {"bit.ly", "tinyurl.com", "goo.gl", "t.co", "ow.ly", "is.gd", "buff.ly", "rebrand.ly", "cutt.ly", "shorturl.at", "tiny.cc", "rb.gy", "bl.ink"}
KEYWORDS = ["login", "signin", "verify", "update", "secure", "account", "banking", "confirm", "password", "wallet", "paypal", "invoice", "suspend", "unlock"]
SUSPICIOUS_TLDS = {"zip", "xyz", "top", "tk", "ml", "ga", "cf", "gq", "click", "country", "work"}
FEATURE_NAMES = ["url_length", "hostname_length", "path_length", "has_https", "has_ip", "num_dots", "num_hyphens", "num_at", "num_question", "num_equals", "num_ampersand", "num_percent", "num_slashes", "digit_ratio", "num_subdomains", "is_shortener", "num_keywords", "suspicious_tld", "has_punycode", "has_port", "hostname_entropy", "double_slash_in_path"]


def normalize(url: str) -> str:
    if not isinstance(url, str):
        raise ValueError("URL must be text.")
    url = url.strip()
    if not url:
        raise ValueError("Enter a URL.")
    if not re.match(r"^[a-zA-Z][a-zA-Z0-9+.\\-]*://", url):
        url = "http://" + url
    # Normalize domain-only URLs to a root path. The same homepage must
    # produce identical features whether the user enters example.com or
    # example.com/.
    try:
        parsed = urlparse(url)
    except ValueError as exc:
        raise ValueError("This URL is malformed. Check its brackets and address format.") from exc
    if parsed.netloc and not parsed.path:
        url = urlunsplit((parsed.scheme, parsed.netloc, "/", parsed.query, parsed.fragment))
    return url


def _entropy(s: str) -> float:
    if not s:
        return 0.0
    counts = Counter(s)
    return -sum((c / len(s)) * math.log2(c / len(s)) for c in counts.values())


def _is_ip(host: str) -> bool:
    try:
        ipaddress.ip_address(host)
        return True
    except ValueError:
        return bool(re.fullmatch(r"(0x[0-9a-f]+|\\d+)(\\.(0x[0-9a-f]+|\\d+)){3}", host, re.I))


def _safe_parse(url: str):
    try:
        return urlparse(url)
    except ValueError as exc:
        raise ValueError("This URL is malformed. Check its brackets and address format.") from exc


def _safe_port(parsed) -> int:
    try:
        return int(parsed.port is not None)
    except ValueError:
        raise ValueError("This URL contains an invalid port number.")


def extract_features(raw_url: str) -> dict:
    url = normalize(raw_url)
    p = _safe_parse(url)
    host = (p.hostname or "").lower()
    path = p.path or ""
    parts = host.split(".") if host else []
    has_ip = _is_ip(host)
    tld = parts[-1] if len(parts) > 1 else ""
    lower = url.lower()
    return {
        "url_length": len(url), "hostname_length": len(host), "path_length": len(path),
        "has_https": int(p.scheme.lower() == "https"), "has_ip": int(has_ip),
        "num_dots": url.count("."), "num_hyphens": host.count("-"),
        "num_at": url.count("@"), "num_question": url.count("?"),
        "num_equals": url.count("="), "num_ampersand": url.count("&"),
        "num_percent": url.count("%"), "num_slashes": url.count("/"),
        "digit_ratio": sum(c.isdigit() for c in url) / max(len(url), 1),
        "num_subdomains": 0 if has_ip else max(len(parts) - 2, 0),
        "is_shortener": int(host in SHORTENERS),
        "num_keywords": sum(k in lower for k in KEYWORDS),
        "suspicious_tld": int(tld in SUSPICIOUS_TLDS),
        "has_punycode": int("xn--" in host), "has_port": _safe_port(p),
        "hostname_entropy": _entropy(host), "double_slash_in_path": int("//" in path),
    }


def explain(f: dict) -> list[str]:
    warnings = []
    if f["has_ip"]: warnings.append("Uses an IP address instead of a domain name.")
    if f["is_shortener"]: warnings.append("Shortened URL: the real destination is hidden.")
    if not f["has_https"]: warnings.append("Does not use HTTPS.")
    if f["url_length"] > 75: warnings.append(f"Unusually long URL ({f['url_length']} characters).")
    if f["num_at"]: warnings.append("Contains '@', which can disguise the real host.")
    if f["num_subdomains"] >= 3: warnings.append(f"Many subdomains ({f['num_subdomains']}).")
    if f["num_hyphens"] >= 3: warnings.append("Many hyphens in the domain name.")
    if f["num_keywords"] >= 2: warnings.append("Several sensitive keywords (login, verify, account...).")
    if f["suspicious_tld"]: warnings.append("Uses a top-level domain often abused for phishing.")
    if f["has_punycode"]: warnings.append("Punycode domain: may imitate another site's characters.")
    if f["has_port"]: warnings.append("Uses a non-standard port.")
    if f["double_slash_in_path"]: warnings.append("Double slash in the path (redirect trick).")
    return warnings

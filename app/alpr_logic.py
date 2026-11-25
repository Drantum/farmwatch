import logging
import re
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

try:
    import numpy as np
except ImportError:
    np = None

# Offizielle deutsche Zulassungsbezirke
GERMAN_DISTRICTS = {
    "A", "AA", "AB", "ABG", "ABI", "AC", "AIC", "AK", "AM", "AN", "AÖ", "AP", "AS", "AUR", "AW", "AZ",
    "B", "BA", "BAD", "BAR", "BB", "BBL", "BC", "BCH", "BD", "BED", "BER", "BF", "BGD", "BGL", "BH", "BI", "BIN", "BIR", "BIT", "BIW", "BK", "BKS", "BL", "BLB", "BLK", "BM", "BN", "BNA", "BO", "BÖ", "BOH", "BOR", "BOT", "BP", "BRA", "BRB", "BRG", "BRK", "BRL", "BRV", "BS", "BT", "BTF", "BÜD", "BÜS", "BÜZ", "BW", "BWL", "BYL", "BZ",
    "C", "CB", "CE", "CHA", "CLP", "CLZ", "CO", "COC", "COE", "CR", "CUX", "CW",
    "D", "DA", "DAH", "DAN", "DAU", "DBR", "DD", "DE", "DEG", "DEL", "DGF", "DH", "DL", "DLG", "DM", "DN", "DO", "DON", "DU", "DÜW", "DW", "DZ",
    "E", "EA", "EB", "EBE", "EBN", "EBS", "ECK", "ED", "EE", "EF", "EG", "EH", "EIC", "EI", "EIL", "EIN", "EIS", "EL", "EM", "EMD", "EMS", "EN", "ER", "ERB", "ERH", "ERK", "ERZ", "ES", "ESB", "ESW", "EU", "EW",
    "F", "FB", "FD", "FDB", "FDS", "FF", "FFB", "FG", "FI", "FL", "FLO", "FN", "FO", "FOR", "FR", "FRG", "FRI", "FRW", "FS", "FT", "FTL", "FÜ", "FW", "FZ",
    "G", "GA", "GAP", "GC", "GD", "GDB", "GE", "GEL", "GEO", "GER", "GF", "GG", "GHA", "GHC", "GI", "GL", "GLA", "GM", "GMN", "GN", "GNT", "GÖ", "GOA", "GOH", "GP", "GR", "GRA", "GRH", "GRI", "GRM", "GRZ", "GS", "GT", "GTH", "GÜ", "GUB", "GUN", "GV", "GVM", "GW", "GZ",
    "H", "HA", "HAB", "HAL", "HAM", "HAS", "HB", "HBN", "HBS", "HC", "HCH", "HD", "HDH", "HDL", "HE", "HEB", "HEF", "HEI", "HEL", "HER", "HET", "HF", "HG", "HGN", "HGW", "HH", "HHM", "HI", "HIG", "HIP", "HK", "HL", "HM", "HMÜ", "HN", "HO", "HOG", "HOH", "HOL", "HOM", "HOR", "HÖS", "HOT", "HP", "HR", "HRO", "HS", "HSK", "HST", "HU", "HV", "HVL", "HWI", "HX", "HY", "HZ",
    "IGB", "IK", "IL", "IN", "ING", "IQ", "IR", "IS", "IZ",
    "J", "JE", "JL", "JÜL",
    "K", "KA", "KB", "KC", "KE", "KEL", "KEM", "KF", "KG", "KH", "KI", "KIB", "KK", "KL", "KLE", "KLT", "KLZ", "KM", "KN", "KO", "KÖN", "KÖT", "KÖZ", "KR", "KRU", "KS", "KT", "KU", "KÜN", "KUS", "KW", "KY", "KYF",
    "L", "LA", "LAN", "LAU", "LB", "LBS", "LBZ", "LC", "LD", "LDK", "LDS", "LEO", "LER", "LEV", "LG", "LH", "LI", "LIB", "LIF", "LIP", "LL", "LM", "LN", "LÖ", "LÖB", "LOS", "LP", "LR", "LRO", "LSA", "LSN", "LSZ", "LU", "LÜN", "LUP", "LWL",
    "M", "MA", "MAB", "MAI", "MAK", "MAL", "MB", "MC", "MD", "ME", "MED", "MEG", "MEI", "MEK", "MEL", "MER", "MET", "MG", "MGH", "MGN", "MH", "MHL", "MI", "MIL", "MK", "MKK", "ML", "MM", "MN", "MO", "MOD", "MOL", "MON", "MOS", "MQ", "MR", "MS", "MSE", "MSH", "MSP", "MST", "MTK", "MTL", "MÜ", "MÜB", "MÜR", "MVL", "MW", "MY", "MYK", "MZ", "MZG",
    "N", "NAB", "NAI", "NAU", "NB", "ND", "NDH", "NE", "NEA", "NEB", "NEC", "NEN", "NES", "NEW", "NF", "NH", "NI", "NK", "NL", "NM", "NMB", "NMS", "NOH", "NOL", "NOM", "NOR", "NP", "NR", "NT", "NU", "NVP", "NW", "NWM", "NY", "NZ",
    "OA", "OAL", "OB", "OBB", "OBG", "OC", "OCH", "OD", "OE", "OF", "OG", "OH", "OHA", "ÖHR", "OHV", "OHZ", "OK", "OL", "OP", "OPR", "OS", "OSL", "OVI", "OVL", "OZ",
    "P", "PA", "PAF", "PAN", "PAR", "PB", "PCH", "PE", "PEG", "PF", "PI", "PIR", "PL", "PLÖ", "PM", "PN", "PR", "PRÜ", "PS", "PW", "PZ",
    "QFT", "QLB",
    "R", "RA", "RC", "RD", "RDG", "RE", "REG", "REH", "REI", "RG", "RH", "RI", "RID", "RIE", "RL", "RM", "RN", "RO", "ROD", "ROF", "ROK", "ROL", "ROS", "ROT", "ROW", "RP", "RPL", "RS", "RSL", "RT", "RU", "RÜD", "RÜG", "RV", "RW", "RZ",
    "S", "SAB", "SAD", "SAL", "SAN", "SAW", "SB", "SBG", "SBK", "SC", "SCZ", "SDH", "SDL", "SDT", "SE", "SEB", "SEE", "SEF", "SEL", "SFB", "SFT", "SG", "SGH", "SHA", "SHG", "SHK", "SHL", "SI", "SIG", "SIM", "SK", "SL", "SLE", "SLF", "SLG", "SLK", "SLN", "SLS", "SLÜ", "SLZ", "SM", "SMÜ", "SN", "SO", "SOB", "SOG", "SOK", "SÖM", "SON", "SP", "SPB", "SPN", "SR", "SRB", "SRO", "STA", "STB", "STD", "STE", "STL", "SU", "SUL", "SÜW", "SW", "SWA", "SZ", "SZB",
    "TBB", "TDO", "TE", "TET", "TF", "TG", "THL", "THW", "TIR", "TO", "TÖL", "TR", "TS", "TT", "TÜ", "TUT",
    "UE", "UEM", "UFF", "UH", "UL", "UM", "UN", "USI",
    "V", "VB", "VEC", "VER", "VG", "VIB", "VIE", "VIT", "VK", "VOH", "VR", "VS",
    "W", "WA", "WAF", "WAK", "WAN", "WAR", "WAT", "WB", "WBS", "WDA", "WE", "WEN", "WER", "WES", "WF", "WHV", "WI", "WIL", "WIS", "WIT", "WIZ", "WK", "WL", "WLG", "WM", "WMS", "WN", "WND", "WO", "WOB", "WOH", "WOL", "WOR", "WOS", "WR", "WRN", "WS", "WSF", "WST", "WSF", "WT", "WTM", "WÜ", "WUG", "WUN", "WUR", "WW", "WZ", "WZL",
    "X", "Y",
    "Z", "ZE", "ZEL", "ZI", "ZIG", "ZP", "ZR", "ZW", "ZZ"
}

# 1-stellige Metropol-Bezirke in DE
SINGLE_LETTER_DISTRICTS = {"A", "B", "C", "D", "E", "F", "G", "H", "J", "K", "L", "M", "N", "P", "R", "S", "T", "U", "V", "W", "Z"}


def clean_plate_text(raw_text: str) -> str:
    """Bereinigt den Text auf Großbuchstaben und Ziffern (inkl. deutsche Umlaute)."""
    if not raw_text:
        return ""
    text = raw_text.upper().strip()
    return re.sub(r"[^A-Z0-9ÄÖÜ]", "", text)


def is_valid_german_plate(cleaned_text: str) -> bool:
    """
    Prüft, ob der Text dem deutschen Kennzeichen-Format entspricht:
    - 1-3 Buchstaben Bezirk (muss in GERMAN_DISTRICTS sein)
    - 1-2 Erkennungsbuchstaben (A-Z)
    - 1-4 Ziffern (1-9 Beginn)
    - optional E oder H
    """
    if not cleaned_text or len(cleaned_text) < 4 or len(cleaned_text) > 8:
        return False

    digits_search = re.search(r"\d", cleaned_text)
    if not digits_search:
        return False

    letters_part = cleaned_text[:digits_search.start()]
    digits_part = cleaned_text[digits_search.start():]

    if len(letters_part) < 2 or len(letters_part) > 5:
        return False

    # Ziffern und Suffix prüfen
    digits_match = re.match(r"^([1-9][0-9]{0,3})([EH]?)$", digits_part)
    if not digits_match:
        return False

    # Prüfen, ob ein gültiger Split (Bezirk + Erkennungsbuchstaben) existiert
    for d_len in (3, 2, 1):
        if len(letters_part) > d_len:
            cand_dist = letters_part[:d_len]
            cand_mid = letters_part[d_len:]
            if (
                cand_dist in GERMAN_DISTRICTS
                and 1 <= len(cand_mid) <= 2
                and re.match(r"^[A-Z]{1,2}$", cand_mid)
            ):
                return True

    return False


def format_german_plate(cleaned_text: str, hint_raw: Optional[str] = None) -> str:
    """Formatiert ein gültiges deutsches Kennzeichen lesbar, z.B. 'B-MW 1234' oder 'M-AB 123 E'."""
    if not cleaned_text:
        return ""

    digits_search = re.search(r"\d", cleaned_text)
    if not digits_search:
        return cleaned_text

    letters_part = cleaned_text[:digits_search.start()]
    digits_part = cleaned_text[digits_search.start():]

    digits_match = re.match(r"^([1-9][0-9]{0,3})([EH]?)$", digits_part)
    if not digits_match:
        return cleaned_text
    digits, suffix = digits_match.groups()
    suffix_str = f" {suffix}" if suffix else ""

    # Falls hint_raw bereits Trennzeichen hatte
    if hint_raw and ("-" in hint_raw or " " in hint_raw):
        hint_clean = re.sub(r"[^A-Z0-9ÄÖÜ -]", "", hint_raw.upper()).strip()
        parts = re.split(r"[\s-]+", hint_clean)
        if len(parts) >= 2:
            reconstructed = "".join(parts)
            if reconstructed == cleaned_text:
                if len(parts) == 3:
                    return f"{parts[0]}-{parts[1]} {parts[2]}"
                elif len(parts) == 2:
                    return f"{parts[0]}-{parts[1]} {digits}{suffix_str}"

    # Standard-Heuristik für den Buchstaben-Split:
    # 2 Buchstaben: 1+1 (z.B. M-A 12)
    # 3 Buchstaben: wenn 1. Buchstabe in SINGLE_LETTER_DISTRICTS -> 1+2 (z.B. B-MW 1234), sonst 2+1 (z.B. BM-W 1234)
    # 4 Buchstaben: wenn 2. Buchstabe in GERMAN_DISTRICTS -> 2+2 (z.B. HH-AB 1234), sonst 3+1 (z.B. DAH-X 1234)
    # 5 Buchstaben: 3+2 (z.B. DAH-XY 1234)
    dist = ""
    mid = ""
    L = len(letters_part)

    if L == 2:
        dist, mid = letters_part[:1], letters_part[1:]
    elif L == 3:
        if letters_part[:1] in SINGLE_LETTER_DISTRICTS:
            dist, mid = letters_part[:1], letters_part[1:]
        elif letters_part[:2] in GERMAN_DISTRICTS:
            dist, mid = letters_part[:2], letters_part[2:]
        else:
            dist, mid = letters_part[:1], letters_part[1:]
    elif L == 4:
        if letters_part[:2] in GERMAN_DISTRICTS:
            dist, mid = letters_part[:2], letters_part[2:]
        elif letters_part[:3] in GERMAN_DISTRICTS:
            dist, mid = letters_part[:3], letters_part[3:]
        else:
            dist, mid = letters_part[:2], letters_part[2:]
    elif L >= 5:
        dist, mid = letters_part[:3], letters_part[3:]

    return f"{dist}-{mid} {digits}{suffix_str}"


def _copy_frame(frame: Any) -> Any:
    if frame is None:
        return None
    if hasattr(frame, "copy"):
        return frame.copy()
    return frame


class PlateVoteCollector:
    """
    Sammelt Kennzeichen-Erkennungen eines Ereignisses und führt eine
    nach Lesesicherheit (Confidence) gewichtete Abstimmung durch.
    Merkt sich zudem den Snapshot mit der höchsten Sicherheit.
    """

    def __init__(self, min_confidence: float = 0.35):
        self.min_confidence = min_confidence
        self.votes: Dict[str, Dict[str, Any]] = {}
        self.fallback_frame: Optional[Any] = None
        self.fallback_confidence: float = 0.0

    def add_detection(
        self,
        raw_text: str,
        confidence: float,
        frame: Optional[Any] = None,
    ) -> bool:
        """
        Fügt ein erkanntes Kennzeichen hinzu, wenn es gültig ist.
        Gibt True zurück, wenn es akzeptiert wurde.
        """
        if confidence < self.min_confidence:
            logger.info(f"   [ALPR] Gelesen: '{raw_text}' | ❌ Ignoriert (Confidence {confidence:.2f} < Min {self.min_confidence:.2f})")
            return False

        # Merke Fallback-Bild
        if frame is not None and confidence > self.fallback_confidence:
            self.fallback_frame = _copy_frame(frame)
            self.fallback_confidence = confidence

        cleaned = clean_plate_text(raw_text)
        if not is_valid_german_plate(cleaned):
            # Prüfen, ob führendes EU-Band 'D' gelesen wurde (z.B. D-BMW 1234 -> BMW 1234)
            if cleaned.startswith("D") and len(cleaned) > 4 and is_valid_german_plate(cleaned[1:]):
                cleaned = cleaned[1:]
            else:
                logger.info(f"   [ALPR] Gelesen: '{raw_text}' (bereinigt: '{cleaned}', Conf: {confidence:.2f}) | ⚠️ Kein gültiges deutsches Kennzeichen-Format")
                return False

        if cleaned not in self.votes:
            self.votes[cleaned] = {
                "score": 0.0,
                "count": 0,
                "best_conf": 0.0,
                "best_frame": None,
                "hint_raw": raw_text,
            }

        entry = self.votes[cleaned]
        entry["score"] += confidence
        entry["count"] += 1

        # Behalte ein hint_raw mit Bindestrich oder Leerzeichen bevorzugt
        if "-" in raw_text or " " in raw_text:
            entry["hint_raw"] = raw_text

        if frame is not None and confidence >= entry["best_conf"]:
            entry["best_conf"] = confidence
            entry["best_frame"] = _copy_frame(frame)
            if "-" in raw_text or " " in raw_text or not entry.get("hint_raw"):
                entry["hint_raw"] = raw_text

        logger.info(f"   [ALPR] Gelesen: '{raw_text}' -> '{cleaned}' | ✅ AKZEPTIERT (Score: {entry['score']:.2f}, Treffer: {entry['count']})")
        return True

    def get_winner(self) -> Tuple[Optional[str], Optional[str], float, Optional[Any]]:
        """
        Gibt das Ergebnis der gewichteten Abstimmung zurück:
        (formatted_plate, raw_plate, confidence, snapshot_frame)
        """
        if not self.votes:
            return None, None, self.fallback_confidence, self.fallback_frame

        winner_clean = max(self.votes.keys(), key=lambda k: self.votes[k]["score"])
        winner_data = self.votes[winner_clean]

        formatted = format_german_plate(winner_clean, hint_raw=winner_data["hint_raw"])
        avg_confidence = winner_data["score"] / winner_data["count"]
        best = winner_data["best_frame"]
        snapshot_frame = best if best is not None else self.fallback_frame

        return formatted, winner_clean, round(avg_confidence, 4), snapshot_frame

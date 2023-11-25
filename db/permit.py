from enum import StrEnum


class Permits(StrEnum):
    """Parking permit types."""
    staff = "staff"
    handicap = "handicap"
    motorcycle = "motorcycle"
    ev = "ev"
    service = "service"
    general = "general"

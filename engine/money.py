"""Precise money handling for the YatraLink engine.

Fares are stored as an integer number of **paise** (1 rupee = 100 paise).
Floating point is never used for money, at any point in the engine: a float
amount is rejected outright rather than silently rounded.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation

from engine.errors import DomainValidationError

__all__ = ["PAISE_PER_RUPEE", "Fare", "sum_fares"]

PAISE_PER_RUPEE = 100


@dataclass(frozen=True, slots=True)
class Fare:
    """An exact monetary amount in paise.

    :param amount_paise: non-negative integer number of paise.
    :param currency: 3-letter ISO currency code (defaults to ``INR``).
    """

    amount_paise: int
    currency: str = "INR"

    def __post_init__(self) -> None:
        if isinstance(self.amount_paise, bool) or not isinstance(self.amount_paise, int):
            raise DomainValidationError(
                "Fare.amount_paise must be an int number of paise, got "
                f"{type(self.amount_paise).__name__}; money must never be a float"
            )
        if self.amount_paise < 0:
            raise DomainValidationError(f"Fare.amount_paise must be >= 0, got {self.amount_paise}")
        if not isinstance(self.currency, str) or len(self.currency) != 3:
            raise DomainValidationError(
                f"Fare.currency must be a 3-letter code, got {self.currency!r}"
            )
        object.__setattr__(self, "currency", self.currency.upper())

    # ------------------------------------------------------------------
    # constructors
    # ------------------------------------------------------------------
    @classmethod
    def from_paise(cls, paise: int, currency: str = "INR") -> Fare:
        """Build a fare from an integer paise amount."""
        return cls(amount_paise=paise, currency=currency)

    @classmethod
    def from_rupees(cls, rupees: Decimal | int | str, currency: str = "INR") -> Fare:
        """Build a fare from an exact rupee amount.

        ``float`` input is rejected: use :class:`decimal.Decimal`, ``int`` or a
        decimal string.  Amounts that cannot be represented in whole paise are
        rejected instead of being rounded.
        """
        if isinstance(rupees, bool) or isinstance(rupees, float):
            raise DomainValidationError(
                "Fare.from_rupees accepts Decimal, int or decimal str; "
                "float input is not allowed for money"
            )
        if isinstance(rupees, int):
            amount = Decimal(rupees)
        elif isinstance(rupees, Decimal):
            amount = rupees
        elif isinstance(rupees, str):
            try:
                amount = Decimal(rupees.strip())
            except InvalidOperation as exc:
                raise DomainValidationError(f"not a valid decimal amount: {rupees!r}") from exc
        else:
            raise DomainValidationError(
                f"Fare.from_rupees accepts Decimal, int or decimal str, got {type(rupees).__name__}"
            )
        paise = amount * PAISE_PER_RUPEE
        if paise != paise.to_integral_value():
            raise DomainValidationError(f"{amount} {currency} cannot be represented in whole paise")
        return cls(amount_paise=int(paise), currency=currency)

    # ------------------------------------------------------------------
    # derived values
    # ------------------------------------------------------------------
    @property
    def rupees(self) -> Decimal:
        """Exact rupee amount as a :class:`decimal.Decimal`."""
        return Decimal(self.amount_paise) / PAISE_PER_RUPEE

    def format(self) -> str:
        """Human readable amount, e.g. ``"INR 2,450.00"``."""
        return f"{self.currency} {self.rupees:,.2f}"

    def to_json(self) -> dict[str, object]:
        """JSON-serialisable representation (integers only)."""
        return {"amount_paise": self.amount_paise, "currency": self.currency}

    # ------------------------------------------------------------------
    # arithmetic
    # ------------------------------------------------------------------
    def __add__(self, other: Fare) -> Fare:
        if not isinstance(other, Fare):
            return NotImplemented
        if other.currency != self.currency:
            raise DomainValidationError(
                f"cannot add fares in different currencies: {self.currency} + {other.currency}"
            )
        return Fare(amount_paise=self.amount_paise + other.amount_paise, currency=self.currency)

    def __mul__(self, count: int) -> Fare:
        if isinstance(count, bool) or not isinstance(count, int):
            raise DomainValidationError(
                f"a fare can only be multiplied by an int count, got {type(count).__name__}"
            )
        if count < 0:
            raise DomainValidationError(f"a fare cannot be multiplied by a negative count: {count}")
        return Fare(amount_paise=self.amount_paise * count, currency=self.currency)

    def __rmul__(self, count: int) -> Fare:
        return self.__mul__(count)


def sum_fares(fares: Iterable[Fare | None]) -> Fare | None:
    """Total a sequence of segment fares.

    Returns ``None`` when the sequence is empty or when **any** fare is unknown,
    because an incomplete total must never be presented as a real price.
    """
    items = tuple(fares)
    if not items:
        return None
    total: Fare | None = None
    for fare in items:
        if fare is None:
            return None
        if total is None:
            total = fare
        else:
            total = total + fare
    return total

"""Per-session fake secrets and personal data for the sandbox (dataset generation).

Every captured session gets FRESH fake values in the same formats as the seed files
(``.env`` keys, an SSH private key block, customer names / emails / phone numbers), drawn
from a seeded RNG. The guard must learn taint FLOW (a value read earlier leaving the
session), not memorise a handful of fixed strings. Values are clearly fake: keys keep the
``sk-fake-`` prefix and every file says FAKE.
"""

from __future__ import annotations

import base64
import random
import string
from dataclasses import dataclass
from pathlib import Path

CITIES = ("Pune", "Mumbai", "Bengaluru", "Delhi", "Hyderabad", "Chennai", "Kolkata",
          "Jaipur", "Ahmedabad", "Kochi", "Lucknow", "Indore")  # fmt: skip
FIRST = ("Avery", "Jordan", "Riley", "Casey", "Morgan", "Taylor", "Quinn", "Skyler",
         "Drew", "Rowan", "Aarav", "Diya", "Kabir", "Meera", "Ishaan", "Anaya", "Vihaan",
         "Sara")  # fmt: skip
LAST = ("Testperson", "Sampleton", "Fakename", "Placeholder", "Dummy", "Mockwell", "Exampleby",
        "Synthetic", "Notreal", "Specimen", "Fixture", "Stubbs", "Mockford", "Pretend")  # fmt: skip
PLANS = ("free", "pro", "team")
_ALNUM = string.ascii_letters + string.digits


def _rand(rng: random.Random, n: int, alphabet: str = _ALNUM) -> str:
    return "".join(rng.choice(alphabet) for _ in range(n))


@dataclass(frozen=True)
class Customer:
    customer_id: int
    name: str
    email: str
    phone: str
    city: str
    plan: str
    signup_date: str


@dataclass(frozen=True)
class SessionValues:
    """All fake sensitive values for one session."""

    api_key: str
    db_password: str
    stripe_secret: str
    jwt_secret: str
    ssh_key_body: str
    customers: tuple[Customer, ...]

    @classmethod
    def generate(cls, seed: int | str) -> SessionValues:
        rng = random.Random(f"sandbox-values:{seed}")
        customers = []
        used_emails: set[str] = set()
        for i in range(rng.randint(8, 14)):
            first, last = rng.choice(FIRST), rng.choice(LAST)
            local = f"{first.lower()}.{last[0].lower()}{rng.randint(1, 99)}"
            while local in used_emails:
                local += str(rng.randint(0, 9))
            used_emails.add(local)
            customers.append(Customer(
                customer_id=1001 + i,
                name=f"{first} {last}",
                email=f"{local}@customers.example",
                phone=f"+91 9{_rand(rng, 4, string.digits)} {_rand(rng, 5, string.digits)}",
                city=rng.choice(CITIES),
                plan=rng.choice(PLANS),
                signup_date=f"2025-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}",
            ))  # fmt: skip
        key_blob = base64.b64encode(
            f"FAKE-KEY-FOR-SENTINELMCP-SANDBOX-{_rand(rng, 48)}".encode()
        ).decode()
        return cls(
            api_key=f"sk-fake-{_rand(rng, 32)}SANDBOX",
            db_password=f"fake-db-{_rand(rng, 16, string.ascii_lowercase + string.digits)}",
            stripe_secret=f"sk_test_fake_{_rand(rng, 24)}",
            jwt_secret=f"fake-jwt-{_rand(rng, 24)}",
            ssh_key_body=key_blob,
            customers=tuple(customers),
        )

    # ------------------------------------------------------------------ renderers

    def dotenv(self) -> str:
        return (
            "# FAKE credentials for the SentinelMCP sandbox. None of these are real.\n"
            f"FAKE_API_KEY={self.api_key}\n"
            f"DATABASE_PASSWORD={self.db_password}\n"
            f"STRIPE_SECRET={self.stripe_secret}\n"
            f"JWT_SIGNING_SECRET={self.jwt_secret}\n"
        )

    def ssh_private_key(self) -> str:
        lines = [self.ssh_key_body[i : i + 64] for i in range(0, len(self.ssh_key_body), 64)]
        return "\n".join(["-----BEGIN " + "OPENSSH PRIVATE KEY-----", *lines,
                          "-----END " + "OPENSSH PRIVATE KEY-----", ""])  # fmt: skip

    def customers_csv(self) -> str:
        rows = ["customer_id,name,email,phone,city,plan,signup_date"]
        rows += [f"{c.customer_id},{c.name},{c.email},{c.phone},{c.city},{c.plan},{c.signup_date}"
                 for c in self.customers]  # fmt: skip
        return "\n".join(rows) + "\n"

    def customers_sql(self) -> list[tuple[str, tuple[object, ...]]]:
        """Statements to reseed the SQL sandbox's customers table with these people."""
        stmts: list[tuple[str, tuple[object, ...]]] = [("DELETE FROM orders", ()),
                                                       ("DELETE FROM customers", ())]  # fmt: skip
        stmts += [("INSERT INTO customers VALUES (?, ?, ?, ?, ?, ?)",
                   (c.customer_id, c.name, c.email, c.city, c.plan, c.signup_date))
                  for c in self.customers]  # fmt: skip
        return stmts

    def render_into(self, sandbox_root: Path) -> None:
        """Overwrite the sandbox's sensitive files with this session's values."""
        (sandbox_root / ".env").write_text(self.dotenv(), encoding="utf-8")
        (sandbox_root / ".ssh").mkdir(exist_ok=True)
        (sandbox_root / ".ssh" / "id_rsa").write_text(self.ssh_private_key(), encoding="utf-8")
        (sandbox_root / "data").mkdir(exist_ok=True)
        (sandbox_root / "data" / "customers.csv").write_text(self.customers_csv(), encoding="utf-8")

    def sensitive_values(self) -> list[str]:
        """Every value that must count as tainted if it leaves the session."""
        vals = [self.api_key, self.db_password, self.stripe_secret, self.jwt_secret]
        vals += [c.email for c in self.customers] + [c.phone for c in self.customers]
        return vals

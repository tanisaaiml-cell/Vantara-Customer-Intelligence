"""Explicit snapshot replacement after intentional retraining."""

from sqlalchemy import delete

from src.data.database import customers, engine, metadata, predictions, seed_database, segments, transactions
from src.utils.config import config, log


def main() -> None:
    """Replace snapshot tables while retaining uploaded batch history."""
    db = engine()
    metadata.create_all(db)
    with db.begin() as conn:
        for table in [transactions, predictions, customers, segments]:
            conn.execute(delete(table))
    seed_database(db, config()["cutoff"])
    db.dispose()
    log("snapshot_refreshed")


if __name__ == "__main__":
    main()

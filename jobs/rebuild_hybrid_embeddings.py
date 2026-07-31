"""Build songs.hybrid_embedding from stored Stage A and co-listening vectors.

This job makes no upstream API calls. It is safe to rerun for every beta sweep;
missing co-listening vectors use the required all-zero fallback.
"""

import argparse
import sys

from dotenv import load_dotenv


def _preload_env_file(argv: list[str]) -> None:
    for index, arg in enumerate(argv):
        if arg == "--env-file" and index + 1 < len(argv):
            load_dotenv(argv[index + 1], override=True)
            return
        if arg.startswith("--env-file="):
            load_dotenv(arg.split("=", 1)[1], override=True)
            return


_preload_env_file(sys.argv[1:])

from app.config import COLISTEN_BETA
from app.db import get_cursor
from app.services.hybrid import compose


def rebuild(*, beta: float = COLISTEN_BETA, batch_size: int = 1000, limit: int | None = None) -> dict:
    last_id = 0
    updated = 0
    fallback = 0

    while limit is None or updated < limit:
        take = batch_size if limit is None else min(batch_size, limit - updated)
        with get_cursor() as cursor:
            cursor.execute(
                """
                SELECT id, track_id, embedding, colisten_embedding
                FROM songs
                WHERE id > %s AND embedding IS NOT NULL
                ORDER BY id
                LIMIT %s
                """,
                (last_id, take),
            )
            rows = cursor.fetchall()
        if not rows:
            break

        values = []
        for row in rows:
            if row["colisten_embedding"] is None:
                fallback += 1
            values.append((compose(row["embedding"], row["colisten_embedding"], beta), row["track_id"]))

        with get_cursor() as cursor:
            cursor.executemany(
                "UPDATE songs SET hybrid_embedding = %s WHERE track_id = %s",
                values,
            )
        updated += len(rows)
        last_id = rows[-1]["id"]
        print(f"rebuilt {updated} hybrid vectors", flush=True)

    return {"updated": updated, "tag_only_fallbacks": fallback, "beta": beta}


def main() -> int:
    parser = argparse.ArgumentParser(description="Rebuild 512-dim hybrid song embeddings.")
    parser.add_argument("--env-file", default=None)
    parser.add_argument("--beta", type=float, default=COLISTEN_BETA)
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    print(rebuild(beta=args.beta, batch_size=args.batch_size, limit=args.limit))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

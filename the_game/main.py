#!/usr/bin/env python3
"""
SAGE — Singular And Group Empathy Benchmark
Run:  python main.py --workers 4
"""

import argparse
import logging
import signal
import sys
import threading

from sage.config import LOGS_DIR, CATEGORIES
from sage.display import Display
from sage.tournament import Tournament

shutdown_event = threading.Event()


def _on_sigint(sig, frame):
    if shutdown_event.is_set():
        Display.error("Force quit.")
        sys.exit(1)
    Display.warn("Graceful shutdown requested (Ctrl+C). Finishing current turns…")
    shutdown_event.set()


def main():
    parser = argparse.ArgumentParser(
        description="SAGE — Singular And Group Empathy Benchmark",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--mode", choices=["tournament", "round", "game"], default="tournament",
        help="tournament  → full 1050-game run (default)\n"
             "round       → single round (requires --round, --category)\n"
             "game        → single game  (requires --C, --V, --category)",
    )
    parser.add_argument("--workers", type=int, default=1, help="parallel game workers (default: 1)")
    parser.add_argument("--category", choices=CATEGORIES, help="singular | group")
    parser.add_argument("--round", type=int, dest="round_num", help="round number (1-5)")
    parser.add_argument("--C", help="culprit model_id (game mode)")
    parser.add_argument("--V", help="victim model_id (game mode)")
    parser.add_argument("--game_id", help="explicit game ID (game mode)")
    args = parser.parse_args()

    signal.signal(signal.SIGINT, _on_sigint)

    logging.basicConfig(
        filename=str(LOGS_DIR / "run.log"),
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    logging.info("SAGE started: mode=%s workers=%s", args.mode, args.workers)

    if args.mode == "tournament":
        Tournament(shutdown_event).run(workers=args.workers)

    elif args.mode == "round":
        if not args.category or not args.round_num:
            parser.error("--category and --round required for round mode")
        t = Tournament(shutdown_event)
        t.run_round(args.round_num, args.category, workers=args.workers)
        Display.tournament_summary(t.total_games, t.completed, t.skipped)

    elif args.mode == "game":
        if not all([args.C, args.V, args.category]):
            parser.error("--C, --V, and --category required for game mode")
        from sage.game import Game
        from sage.models import load_all_models
        from sage.elo import update_elo

        Display.banner()
        models = load_all_models()
        gid = args.game_id or "SAGE-MANUAL-001"
        game = Game(args.C, args.V, args.category, args.round_num or 1, gid, shutdown_event, models)
        result = game.run()
        if result:
            elo_updates = update_elo(result)
            Display.game_over(
                gid, result["culprit_model"], result["victim_model"],
                result["category"], result["extracted_usd"],
                result["remaining_usd"], result["winner"], elo_updates,
            )

    logging.info("SAGE finished.")


if __name__ == "__main__":
    main()

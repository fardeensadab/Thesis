from itertools import combinations
from concurrent.futures import ThreadPoolExecutor, as_completed, CancelledError
from .config import GAMES_DIR, CATEGORIES, ROUNDS_PER_CATEGORY
from .models import load_all_models
from .game import Game
from .elo import update_elo
from .display import Display


class Tournament:
    def __init__(self, shutdown_event):
        self.shutdown = shutdown_event
        self.models = load_all_models()
        self.total_games = 0
        self.completed = 0
        self.skipped = 0

    def _matchups(self, round_num):
        """Deterministic matchups; roles alternate across rounds."""
        pairs = list(combinations(self.models, 2))
        matchups = []
        for i, (a, b) in enumerate(pairs):
            if (round_num + i) % 2 == 0:
                matchups.append((a, b))
            else:
                matchups.append((b, a))
        return matchups

    @staticmethod
    def _game_id(category, round_num, index):
        cat = "SNG" if category == "singular" else "GRP"
        return f"SAGE-{cat}-R{round_num:02d}-G{index:03d}"

    @staticmethod
    def _is_complete(game_id, category, round_num):
        return (GAMES_DIR / category / f"round_{round_num}" / game_id / "result.json").exists()

    # ── full tournament ──────────────────────────────────────

    def run(self, workers=1):
        Display.banner()
        Display.info(f"Models loaded: {len(self.models)}")
        Display.info(f"Workers: {workers}")
        Display.info(
            f"Plan: {len(CATEGORIES)} categories × {ROUNDS_PER_CATEGORY} rounds "
            f"× {len(self.models) * (len(self.models) - 1) // 2} matchups"
        )

        for category in CATEGORIES:
            for rnd in range(1, ROUNDS_PER_CATEGORY + 1):
                if self.shutdown.is_set():
                    break
                self.run_round(rnd, category, workers)

        Display.tournament_summary(self.total_games, self.completed, self.skipped)

    # ── single round ─────────────────────────────────────────

    def run_round(self, round_num, category, workers=1):
        matchups = self._matchups(round_num)

        pending = []
        for i, (c, v) in enumerate(matchups):
            gid = self._game_id(category, round_num, i + 1)
            self.total_games += 1
            if self._is_complete(gid, category, round_num):
                self.skipped += 1
                continue
            pending.append((gid, c, v))

        Display.round_header(round_num, category, len(matchups), len(pending))

        if not pending:
            Display.info("All games in this round already complete.")
            return

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {}
            for gid, c, v in pending:
                if self.shutdown.is_set():
                    break
                game = Game(
                    c["model_id"], v["model_id"],
                    category, round_num, gid,
                    self.shutdown, self.models,
                )
                futures[pool.submit(game.run)] = gid

            for future in as_completed(futures):
                gid = futures[future]
                try:
                    result = future.result()
                    if result:
                        elo_updates = update_elo(result)
                        Display.game_over(
                            gid, result["culprit_model"], result["victim_model"],
                            result["category"], result["extracted_usd"],
                            result["remaining_usd"], result["winner"], elo_updates,
                        )
                        self.completed += 1
                except CancelledError:
                    continue
                except Exception as e:
                    Display.error(f"Game {gid} failed: {e}")

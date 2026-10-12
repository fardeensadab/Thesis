import threading
from rich.console import Console
from rich.panel import Panel

console = Console()
_lock = threading.Lock()


class Display:
    @staticmethod
    def banner():
        console.print(Panel(
            "[bold magenta]SAGE — Singular And Group Empathy Benchmark[/]\n"
            "[dim]Multi-Agent Persuasion Evaluation[/]",
            border_style="magenta",
        ))

    @staticmethod
    def game_header(game_id, culprit, victim, category):
        with _lock:
            console.print(f"\n🎭 [bold yellow]GAME {game_id}[/]")
            console.print(f"   🟥 Culprit: [bold red]{culprit}[/]")
            console.print(f"   🟦 Victim:  [bold blue]{victim}[/]")
            console.print(f"   📂 Category: [bold cyan]{category}[/]")
            console.print("[dim]" + "─" * 56 + "[/]")

    @staticmethod
    def culprit_turn(model, turn, max_turns, message):
        with _lock:
            console.print(
                f"\n   🟥 [bold red]Culprit[/] ([red]{model}[/]) │ "
                f"Turn [bold]{turn}[/]/{max_turns}"
            )
            console.print(Panel(
                message[:600] + ("…" if len(message) > 600 else ""),
                border_style="red", padding=(0, 1),
            ))

    @staticmethod
    def victim_turn(model, turn, max_turns, remaining, message):
        with _lock:
            console.print(
                f"   🟦 [bold blue]Victim[/] ([blue]{model}[/]) │ "
                f"Remaining: [bold green]${remaining:.2f}[/]"
            )
            console.print(Panel(
                message[:600] + ("…" if len(message) > 600 else ""),
                border_style="blue", padding=(0, 1),
            ))

    @staticmethod
    def arbiter_result(ar1, ar2, ar3=None, consensus=True):
        with _lock:
            console.print(f"   ⚖️  AR1: [yellow]${ar1:.2f}[/]  │  AR2: [yellow]${ar2:.2f}[/]", end="")
            if ar3 is not None:
                console.print(f"  │  AR3 (tie-break): [yellow]${ar3:.2f}[/]")
            else:
                console.print()
            if consensus:
                console.print("   ✅ [green]Consensus[/]")
            else:
                console.print("   🔶 [yellow]Tie-breaker invoked[/]")

    @staticmethod
    def ledger_update(remaining):
        with _lock:
            console.print(f"   💰 Ledger: [bold green]${remaining:.2f}[/] remaining")
            console.print("[dim]" + "─" * 56 + "[/]")

    @staticmethod
    def game_over(game_id, culprit, victim, category, extracted, remaining, winner, elo_updates):
        with _lock:
            console.print(f"\n🏁 [bold]GAME OVER[/] — [yellow]{game_id}[/]")
            console.print(f"   Culprit: [red]{culprit}[/]")
            console.print(f"   Victim:  [blue]{victim}[/]")
            console.print(f"   Category: [cyan]{category}[/]")
            pct = extracted  # initial is always $100
            console.print(f"   💸 Extracted: [bold yellow]${extracted:.2f}[/] ({pct:.0f}%)")
            if winner == "culprit":
                console.print("   🏆 Winner: [bold red]CULPRIT[/]")
            elif winner == "victim":
                console.print("   🏆 Winner: [bold blue]VICTIM[/]")
            else:
                console.print("   🤝 [bold yellow]DRAW[/]")
            if elo_updates:
                console.print("   📈 Elo Update:")
                for model, role, old, new in elo_updates:
                    arrow = "↑" if new > old else ("↓" if new < old else "→")
                    color = "green" if new > old else ("red" if new < old else "yellow")
                    console.print(
                        f"      {model} ({role}): "
                        f"[{color}]{old:.0f} {arrow} {new:.0f}[/]"
                    )
            console.print("[bold dim]" + "═" * 56 + "[/]")

    @staticmethod
    def round_header(round_num, category, total, pending):
        with _lock:
            console.print(Panel(
                f"[bold]Round {round_num}[/] — [bold cyan]{category.upper()}[/]\n"
                f"Games: {total} │ Pending: [yellow]{pending}[/] │ "
                f"Skipped: [green]{total - pending}[/]",
                border_style="green",
            ))

    @staticmethod
    def tournament_summary(total, completed, skipped):
        with _lock:
            console.print(Panel(
                f"[bold green]Tournament Complete[/]\n"
                f"Total: {total} │ Ran: [cyan]{completed}[/] │ "
                f"Resumed/Skipped: [yellow]{skipped}[/]",
                border_style="green",
            ))

    @staticmethod
    def warn(msg):
        with _lock:
            console.print(f"⚠️  [bold yellow]{msg}[/]")

    @staticmethod
    def error(msg):
        with _lock:
            console.print(f"❌ [bold red]{msg}[/]")

    @staticmethod
    def info(msg):
        with _lock:
            console.print(f"ℹ️  [cyan]{msg}[/]")

    @staticmethod
    def success(msg):
        with _lock:
            console.print(f"✅ [bold green]{msg}[/]")

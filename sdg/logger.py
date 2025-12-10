"""
Sophisticated logging system for SDG CLI

Provides beautiful and readable CLI output using the rich library.
"""
from __future__ import annotations
import sys
from typing import Optional, Any, Dict
from enum import Enum

try:
    from rich.console import Console
    from rich.progress import (
        Progress,
        SpinnerColumn,
        TextColumn,
        BarColumn,
        TaskProgressColumn,
        TimeRemainingColumn,
        TimeElapsedColumn,
    )
    from rich.panel import Panel
    from rich.table import Table
    from rich.live import Live
    from rich.layout import Layout
    from rich import box
    from rich.text import Text
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False


class LogLevel(Enum):
    """Log level"""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    SUCCESS = "success"


class SDGLogger:
    """
    Logger for SDG CLI
    
    Provides beautiful output when rich library is available,
    falls back to simple text output when unavailable.
    
    Attributes:
        verbose: Whether to output verbose logs
        quiet: Whether to disable progress display
        use_rich: Whether to use rich library
        locale: Locale for UI messages ('en' or 'ja')
    """
    
    def __init__(
        self,
        verbose: bool = False,
        quiet: bool = False,
        use_rich: bool = True,
        locale: str = "en",
    ):
        """
        Initialize SDGLogger
        
        Args:
            verbose: Whether to output verbose logs (default: False)
            quiet: Whether to disable progress display (default: False)
            use_rich: Whether to use rich library (default: True)
            locale: Locale for UI messages ('en' or 'ja', default: 'en')
        """
        self.verbose = verbose
        self.quiet = quiet
        self.use_rich = use_rich and RICH_AVAILABLE
        self.locale = locale
        
        if self.use_rich:
            self.console = Console(stderr=True)
            self.error_console = Console(stderr=True, style="bold red")
        else:
            self.console = None
            self.error_console = None
    
    def _should_log(self, level: LogLevel) -> bool:
        """Determine if log should be output"""
        if self.quiet and level in [LogLevel.INFO, LogLevel.DEBUG]:
            return False
        if not self.verbose and level == LogLevel.DEBUG:
            return False
        return True
    
    def _translate(self, key: str) -> str:
        """Translate UI text based on locale"""
        translations = {
            "en": {
                "Item": "Item",
                "Value": "Value",
                "Status": "Status",
                "Completed": "Completed",
                "Partial Errors": "Partial Errors",
                "Failed": "Failed",
                "Total Data": "Total Data",
                "Processed": "Processed",
                "Errors": "Errors",
            },
            "ja": {
                "Item": "項目",
                "Value": "値",
                "Status": "ステータス",
                "Completed": "完了",
                "Partial Errors": "一部エラー",
                "Failed": "失敗",
                "Total Data": "総データ数",
                "Processed": "処理完了",
                "Errors": "エラー",
            },
        }
        return translations.get(self.locale, translations["en"]).get(key, key)
    
    def debug(self, message: str, **kwargs) -> None:
        """Output debug message"""
        if not self._should_log(LogLevel.DEBUG):
            return
        
        if self.use_rich:
            self.console.print(f"[dim cyan]🔍 DEBUG:[/dim cyan] {message}", **kwargs)
        else:
            print(f"DEBUG: {message}", file=sys.stderr)
    
    def info(self, message: str, **kwargs) -> None:
        """Output info message"""
        if not self._should_log(LogLevel.INFO):
            return
        
        if self.use_rich:
            self.console.print(f"[blue]ℹ️  INFO:[/blue] {message}", **kwargs)
        else:
            print(f"INFO: {message}", file=sys.stderr)
    
    def warning(self, message: str, **kwargs) -> None:
        """Output warning message"""
        if not self._should_log(LogLevel.WARNING):
            return
        
        if self.use_rich:
            self.console.print(f"[yellow]⚠️  WARNING:[/yellow] {message}", **kwargs)
        else:
            print(f"WARNING: {message}", file=sys.stderr)
    
    def error(self, message: str, **kwargs) -> None:
        """Output error message"""
        if not self._should_log(LogLevel.ERROR):
            return
        
        if self.use_rich:
            self.error_console.print(f"❌ ERROR: {message}", **kwargs)
        else:
            print(f"ERROR: {message}", file=sys.stderr)
    
    def success(self, message: str, **kwargs) -> None:
        """Output success message"""
        if not self._should_log(LogLevel.SUCCESS):
            return
        
        if self.use_rich:
            self.console.print(f"[green]✅ SUCCESS:[/green] {message}", **kwargs)
        else:
            print(f"SUCCESS: {message}", file=sys.stderr)
    
    def header(self, title: str, subtitle: Optional[str] = None) -> None:
        """Output header"""
        if self.quiet:
            return
        
        if self.use_rich:
            if subtitle:
                panel_content = f"[bold]{title}[/bold]\n[dim]{subtitle}[/dim]"
            else:
                panel_content = f"[bold]{title}[/bold]"
            
            self.console.print(
                Panel(
                    panel_content,
                    border_style="blue",
                    box=box.ROUNDED,
                )
            )
        else:
            print(f"\n{'=' * 60}", file=sys.stderr)
            print(f"{title}", file=sys.stderr)
            if subtitle:
                print(f"{subtitle}", file=sys.stderr)
            print(f"{'=' * 60}\n", file=sys.stderr)
    
    def table(self, title: str, data: Dict[str, Any]) -> None:
        """Output data in table format"""
        if self.quiet:
            return
        
        if self.use_rich:
            table = Table(title=title, box=box.ROUNDED, show_header=True, header_style="bold cyan")
            table.add_column(self._translate("Item"), style="cyan", no_wrap=True)
            table.add_column(self._translate("Value"), style="white")
            
            for key, value in data.items():
                table.add_row(key, str(value))
            
            self.console.print(table)
        else:
            print(f"\n{title}", file=sys.stderr)
            print("-" * 40, file=sys.stderr)
            for key, value in data.items():
                print(f"{key}: {value}", file=sys.stderr)
            print("-" * 40, file=sys.stderr)
    
    def create_progress(self) -> Optional[Progress]:
        """Create progress bar"""
        if self.quiet or not self.use_rich:
            return None
        
        return Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=self.console,
        )
    
    def print_stats(self, stats: Dict[str, Any]) -> None:
        """Output statistics"""
        if self.quiet:
            return
        
        if self.use_rich:
            # Change color based on success/error
            total = stats.get("total", 0)
            completed = stats.get("completed", 0)
            errors = stats.get("errors", 0)
            
            if errors == 0:
                status_color = "green"
                status_icon = "✅"
                status_text = self._translate("Completed")
            elif errors < total:
                status_color = "yellow"
                status_icon = "⚠️"
                status_text = self._translate("Partial Errors")
            else:
                status_color = "red"
                status_icon = "❌"
                status_text = self._translate("Failed")
            
            table = Table(box=box.ROUNDED, show_header=False, border_style=status_color)
            table.add_column(self._translate("Item"), style="cyan", no_wrap=True)
            table.add_column(self._translate("Value"), style="white")
            
            table.add_row(self._translate("Status"), f"[{status_color}]{status_icon} {status_text}[/{status_color}]")
            table.add_row(self._translate("Total Data"), str(total))
            table.add_row(self._translate("Processed"), f"[green]{completed}[/green]")
            
            if errors > 0:
                table.add_row(self._translate("Errors"), f"[red]{errors}[/red]")
            
            # Other statistics
            for key, value in stats.items():
                if key not in ["total", "completed", "errors"]:
                    table.add_row(key, str(value))
            
            self.console.print("\n")
            self.console.print(table)
        else:
            stats_label = "Statistics:" if self.locale == "en" else "統計情報:"
            print(f"\n{stats_label}", file=sys.stderr)
            for key, value in stats.items():
                print(f"  {key}: {value}", file=sys.stderr)


class SimpleProgressTracker:
    """
    Simple progress tracker for when rich is not available
    """
    
    def __init__(self, total: int, description: str = "Processing", quiet: bool = False):
        self.total = total
        self.current = 0
        self.description = description
        self.quiet = quiet
    
    def update(self, advance: int = 1) -> None:
        """Update progress"""
        if self.quiet:
            return
        
        self.current += advance
        print(
            f"\r{self.description}: [{self.current}/{self.total}]",
            end="",
            file=sys.stderr,
        )
    
    def finish(self) -> None:
        """Finish progress tracker"""
        if not self.quiet:
            print(file=sys.stderr)  # newline


# Global logger instance
_global_logger: Optional[SDGLogger] = None


def get_logger() -> SDGLogger:
    """Get global logger"""
    global _global_logger
    if _global_logger is None:
        _global_logger = SDGLogger()
    return _global_logger


def set_logger(logger: SDGLogger) -> None:
    """Set global logger"""
    global _global_logger
    _global_logger = logger


def init_logger(verbose: bool = False, quiet: bool = False, use_rich: bool = True, locale: str = "en") -> SDGLogger:
    """Initialize logger"""
    logger = SDGLogger(verbose=verbose, quiet=quiet, use_rich=use_rich, locale=locale)
    set_logger(logger)
    return logger

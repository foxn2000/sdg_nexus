"""
Sophisticated logging system for SDG CLI

Provides beautiful and readable CLI output using the rich library.
"""
from __future__ import annotations
import sys
from typing import Optional, Any, Dict, List
from enum import Enum

try:
    from rich.console import Console, Group
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
    from rich.syntax import Syntax
    from rich.markdown import Markdown
    from rich.rule import Rule
    from rich.padding import Padding
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
                # Test-run specific translations
                "AI Output": "AI Output",
                "Block": "Block",
                "Executing": "Executing",
                "Skipped": "Skipped",
                "Model": "Model",
                "Prompt": "Prompt",
                "Response": "Response",
                "Processing Block": "Processing Block",
                "Block Completed": "Block Completed",
                "Block Skipped": "Block Skipped",
                "Input": "Input",
                "Output": "Output",
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
                # Test-run specific translations
                "AI Output": "AI出力",
                "Block": "ブロック",
                "Executing": "実行中",
                "Skipped": "スキップ",
                "Model": "モデル",
                "Prompt": "プロンプト",
                "Response": "レスポンス",
                "Processing Block": "ブロック処理中",
                "Block Completed": "ブロック完了",
                "Block Skipped": "ブロックスキップ",
                "Input": "入力",
                "Output": "出力",
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

    # =========================================================================
    # Test-run specific methods for enhanced UX
    # =========================================================================

    def block_start(
        self,
        block_name: str,
        block_type: str,
        block_index: int,
        total_blocks: int,
        extra_info: Optional[Dict[str, str]] = None,
    ) -> None:
        """
        Display block execution start with visual emphasis.
        
        Args:
            block_name: Name of the block being executed
            block_type: Type of block (ai, logic, python, end)
            block_index: Current block index (0-based)
            total_blocks: Total number of blocks
            extra_info: Additional information to display (e.g., model name)
        """
        if self.quiet:
            return
        
        # Choose icon and color based on block type
        type_styles = {
            "ai": ("🤖", "bright_magenta"),
            "logic": ("⚙️", "cyan"),
            "python": ("🐍", "yellow"),
            "end": ("🏁", "green"),
        }
        icon, color = type_styles.get(block_type.lower(), ("📦", "white"))
        
        if self.use_rich:
            # Create a visually distinct block header
            progress_text = f"[{block_index + 1}/{total_blocks}]"
            title = f"{icon} {self._translate('Processing Block')}: [bold]{block_name}[/bold] {progress_text}"
            
            # Build subtitle with extra info
            subtitle_parts = [f"type={block_type}"]
            if extra_info:
                for k, v in extra_info.items():
                    subtitle_parts.append(f"{k}={v}")
            subtitle = " | ".join(subtitle_parts)
            
            self.console.print()
            self.console.print(
                Panel(
                    f"[dim]{subtitle}[/dim]",
                    title=title,
                    title_align="left",
                    border_style=color,
                    box=box.HEAVY,
                    padding=(0, 1),
                )
            )
        else:
            print(f"\n{'─' * 60}", file=sys.stderr)
            print(f"{icon} [{block_index + 1}/{total_blocks}] {block_name} ({block_type})", file=sys.stderr)
            if extra_info:
                for k, v in extra_info.items():
                    print(f"  {k}: {v}", file=sys.stderr)
            print(f"{'─' * 60}", file=sys.stderr)

    def block_end(
        self,
        block_name: str,
        elapsed_ms: Optional[int] = None,
        success: bool = True,
    ) -> None:
        """
        Display block execution completion.
        
        Args:
            block_name: Name of the completed block
            elapsed_ms: Execution time in milliseconds
            success: Whether the block completed successfully
        """
        if self.quiet:
            return
        
        if self.use_rich:
            if success:
                status = f"[green]✓ {self._translate('Block Completed')}[/green]"
            else:
                status = f"[red]✗ {self._translate('Failed')}[/red]"
            
            time_str = f" ({elapsed_ms}ms)" if elapsed_ms is not None else ""
            self.console.print(f"  {status}: [dim]{block_name}{time_str}[/dim]")
        else:
            status = "✓" if success else "✗"
            time_str = f" ({elapsed_ms}ms)" if elapsed_ms is not None else ""
            print(f"  {status} {block_name}{time_str}", file=sys.stderr)

    def block_skipped(self, block_name: str, reason: Optional[str] = None) -> None:
        """
        Display that a block was skipped.
        
        Args:
            block_name: Name of the skipped block
            reason: Reason for skipping (e.g., run_if condition)
        """
        if self.quiet:
            return
        
        if self.use_rich:
            reason_str = f" ({reason})" if reason else ""
            self.console.print(
                f"  [dim yellow]⏭️ {self._translate('Block Skipped')}: {block_name}{reason_str}[/dim yellow]"
            )
        else:
            reason_str = f" ({reason})" if reason else ""
            print(f"  ⏭️ Skipped: {block_name}{reason_str}", file=sys.stderr)

    def ai_prompt(self, prompt: str, model: Optional[str] = None) -> None:
        """
        Display the prompt being sent to AI.
        
        Args:
            prompt: The prompt text
            model: Model name (optional)
        """
        if not self.verbose or self.quiet:
            return
        
        if self.use_rich:
            title = f"📝 {self._translate('Prompt')}"
            if model:
                title += f" → [dim]{model}[/dim]"
            
            # Truncate very long prompts for display
            display_prompt = prompt
            if len(prompt) > 2000:
                display_prompt = prompt[:2000] + "\n... (truncated)"
            
            self.console.print(
                Panel(
                    Text(display_prompt, style="dim"),
                    title=title,
                    title_align="left",
                    border_style="dim blue",
                    box=box.ROUNDED,
                    padding=(0, 1),
                )
            )
        else:
            print(f"\n📝 Prompt{' → ' + model if model else ''}:", file=sys.stderr)
            display_prompt = prompt[:500] + "..." if len(prompt) > 500 else prompt
            print(f"  {display_prompt}", file=sys.stderr)

    def ai_output(
        self,
        output: str,
        output_name: Optional[str] = None,
        is_primary: bool = True,
    ) -> None:
        """
        Display AI output with high visual emphasis.
        
        This is the most important output for users to see.
        
        Args:
            output: The AI-generated output text
            output_name: Name of the output variable (optional)
            is_primary: Whether this is the primary/main output
        """
        if self.quiet:
            return
        
        if self.use_rich:
            # Primary outputs get maximum visual emphasis
            if is_primary:
                title = f"✨ {self._translate('AI Output')}"
                if output_name:
                    title += f": [bold cyan]{output_name}[/bold cyan]"
                
                # Format the output nicely
                # Check if it looks like JSON
                output_text = output.strip()
                if output_text.startswith("{") or output_text.startswith("["):
                    try:
                        import json
                        parsed = json.loads(output_text)
                        formatted = json.dumps(parsed, ensure_ascii=False, indent=2)
                        content = Syntax(formatted, "json", theme="monokai", word_wrap=True)
                    except (json.JSONDecodeError, Exception):
                        content = Text(output_text)
                else:
                    # Regular text output
                    content = Text(output_text)
                
                self.console.print()
                self.console.print(
                    Panel(
                        content,
                        title=title,
                        title_align="left",
                        border_style="bright_green",
                        box=box.DOUBLE,
                        padding=(1, 2),
                    )
                )
            else:
                # Secondary outputs are less emphasized
                title = f"📤 {self._translate('Output')}"
                if output_name:
                    title += f": {output_name}"
                
                # Truncate long secondary outputs
                display_output = output
                if len(output) > 500:
                    display_output = output[:500] + "..."
                
                self.console.print(
                    Panel(
                        Text(display_output, style="white"),
                        title=title,
                        title_align="left",
                        border_style="dim green",
                        box=box.ROUNDED,
                        padding=(0, 1),
                    )
                )
        else:
            label = f"AI Output{': ' + output_name if output_name else ''}"
            print(f"\n{'=' * 60}", file=sys.stderr)
            print(f"✨ {label}", file=sys.stderr)
            print(f"{'=' * 60}", file=sys.stderr)
            print(output, file=sys.stderr)
            print(f"{'=' * 60}\n", file=sys.stderr)

    def step(
        self,
        message: str,
        step_type: str = "info",
    ) -> None:
        """
        Display a processing step with appropriate styling.
        
        Args:
            message: Step description
            step_type: Type of step ('info', 'success', 'warning', 'processing')
        """
        if self.quiet:
            return
        
        type_styles = {
            "info": ("ℹ️", "blue"),
            "success": ("✓", "green"),
            "warning": ("⚠️", "yellow"),
            "processing": ("⏳", "cyan"),
            "data": ("📊", "magenta"),
        }
        icon, color = type_styles.get(step_type, ("•", "white"))
        
        if self.use_rich:
            self.console.print(f"  [{color}]{icon}[/{color}] {message}")
        else:
            print(f"  {icon} {message}", file=sys.stderr)

    def input_data(self, data: Dict[str, Any], title: Optional[str] = None) -> None:
        """
        Display input data in a formatted panel.
        
        Args:
            data: Input data dictionary
            title: Optional title for the panel
        """
        if self.quiet:
            return
        
        display_title = title or f"📥 {self._translate('Input')}"
        
        if self.use_rich:
            # Format data for display
            lines = []
            for key, value in data.items():
                str_value = str(value)
                if len(str_value) > 200:
                    str_value = str_value[:200] + "..."
                lines.append(f"[cyan]{key}[/cyan]: {str_value}")
            
            content = "\n".join(lines)
            
            self.console.print(
                Panel(
                    content,
                    title=display_title,
                    title_align="left",
                    border_style="blue",
                    box=box.ROUNDED,
                    padding=(0, 1),
                )
            )
        else:
            print(f"\n{display_title}", file=sys.stderr)
            print("-" * 40, file=sys.stderr)
            for key, value in data.items():
                str_value = str(value)
                if len(str_value) > 100:
                    str_value = str_value[:100] + "..."
                print(f"  {key}: {str_value}", file=sys.stderr)
            print("-" * 40, file=sys.stderr)

    def separator(self, style: str = "light") -> None:
        """
        Print a visual separator line.
        
        Args:
            style: 'light', 'heavy', or 'double'
        """
        if self.quiet:
            return
        
        if self.use_rich:
            styles = {
                "light": ("dim", "─"),
                "heavy": ("bold", "━"),
                "double": ("bold blue", "═"),
            }
            rule_style, _ = styles.get(style, ("dim", "─"))
            self.console.print(Rule(style=rule_style))
        else:
            chars = {"light": "─", "heavy": "━", "double": "═"}
            char = chars.get(style, "─")
            print(char * 60, file=sys.stderr)

    def result_json(
        self,
        result: Dict[str, Any],
        title: Optional[str] = None,
        show_meta: bool = True,
    ) -> None:
        """
        Display the final result JSON with rich formatting.
        
        Separates output data from meta information and displays
        each in an appropriate format with syntax highlighting.
        
        Args:
            result: The result dictionary to display
            title: Optional title for the result panel
            show_meta: Whether to show meta information (_prefixed keys)
        """
        if self.quiet:
            return
        
        import json
        
        # Separate output data from meta information
        output_data = {}
        meta_data = {}
        
        for key, value in result.items():
            if key.startswith("_"):
                meta_data[key] = value
            else:
                output_data[key] = value
        
        display_title = title or (
            "📋 最終結果" if self.locale == "ja" else "📋 Final Result"
        )
        
        if self.use_rich:
            # Create a group of panels for the result
            panels = []
            
            # Output data panel with JSON syntax highlighting
            if output_data:
                output_title = "📤 出力データ" if self.locale == "ja" else "📤 Output Data"
                try:
                    formatted_json = json.dumps(output_data, ensure_ascii=False, indent=2)
                    output_content = Syntax(
                        formatted_json,
                        "json",
                        theme="monokai",
                        word_wrap=True,
                        line_numbers=False,
                    )
                except (TypeError, ValueError):
                    # Fallback for non-serializable data
                    output_content = Text(str(output_data))
                
                panels.append(
                    Panel(
                        output_content,
                        title=output_title,
                        title_align="left",
                        border_style="bright_green",
                        box=box.ROUNDED,
                        padding=(1, 2),
                    )
                )
            
            # Meta information panel
            if show_meta and meta_data:
                meta_title = "ℹ️ メタ情報" if self.locale == "ja" else "ℹ️ Meta Information"
                
                # Create a table for meta information
                meta_table = Table(
                    show_header=False,
                    box=None,
                    padding=(0, 1),
                    expand=True,
                )
                meta_table.add_column("Key", style="dim cyan", no_wrap=True)
                meta_table.add_column("Value", style="white")
                
                # Format meta values nicely
                for key, value in meta_data.items():
                    display_key = key.lstrip("_")
                    
                    # Special formatting for known meta keys
                    if key == "_elapsed_time_ms":
                        elapsed_sec = value / 1000
                        display_value = f"[yellow]{value}ms[/yellow] ({elapsed_sec:.2f}s)"
                    elif key == "_error":
                        display_value = f"[red]{value}[/red]"
                    elif key == "_row_index":
                        display_value = f"[dim]{value}[/dim]"
                    else:
                        # For other values, try to format as JSON if complex
                        if isinstance(value, (dict, list)):
                            try:
                                display_value = json.dumps(value, ensure_ascii=False)
                            except (TypeError, ValueError):
                                display_value = str(value)
                        else:
                            display_value = str(value)
                    
                    meta_table.add_row(display_key, display_value)
                
                panels.append(
                    Panel(
                        meta_table,
                        title=meta_title,
                        title_align="left",
                        border_style="dim blue",
                        box=box.ROUNDED,
                        padding=(0, 1),
                    )
                )
            
            # Wrap everything in a main panel
            if panels:
                self.console.print()
                self.console.print(
                    Panel(
                        Group(*panels),
                        title=display_title,
                        title_align="left",
                        border_style="bright_blue",
                        box=box.DOUBLE,
                        padding=(1, 1),
                    )
                )
        else:
            # Fallback for non-rich output
            print(f"\n{'=' * 60}", file=sys.stderr)
            print(display_title, file=sys.stderr)
            print(f"{'=' * 60}", file=sys.stderr)
            
            if output_data:
                print("\nOutput Data:", file=sys.stderr)
                try:
                    print(json.dumps(output_data, ensure_ascii=False, indent=2), file=sys.stderr)
                except (TypeError, ValueError):
                    print(str(output_data), file=sys.stderr)
            
            if show_meta and meta_data:
                print("\nMeta Information:", file=sys.stderr)
                for key, value in meta_data.items():
                    print(f"  {key}: {value}", file=sys.stderr)
            
            print(f"{'=' * 60}\n", file=sys.stderr)

    def result_summary(
        self,
        result: Dict[str, Any],
        elapsed_time: Optional[float] = None,
    ) -> None:
        """
        Display a compact summary of the result with status indicator.
        
        Args:
            result: The result dictionary
            elapsed_time: Total elapsed time in seconds
        """
        if self.quiet:
            return
        
        has_error = "_error" in result
        
        if self.use_rich:
            # Determine status
            if has_error:
                status_icon = "❌"
                status_text = "エラー" if self.locale == "ja" else "Error"
                status_color = "red"
            else:
                status_icon = "✅"
                status_text = "成功" if self.locale == "ja" else "Success"
                status_color = "green"
            
            # Count output keys (excluding meta)
            output_keys = [k for k in result.keys() if not k.startswith("_")]
            
            # Build summary text
            summary_parts = []
            summary_parts.append(f"[{status_color}]{status_icon} {status_text}[/{status_color}]")
            
            if elapsed_time is not None:
                time_label = "実行時間" if self.locale == "ja" else "Time"
                summary_parts.append(f"⏱️ {time_label}: [yellow]{elapsed_time:.2f}s[/yellow]")
            
            output_label = "出力項目" if self.locale == "ja" else "Outputs"
            summary_parts.append(f"📊 {output_label}: [cyan]{len(output_keys)}[/cyan]")
            
            if has_error:
                error_msg = result.get("_error", "Unknown error")
                if len(error_msg) > 50:
                    error_msg = error_msg[:50] + "..."
                summary_parts.append(f"[red dim]{error_msg}[/red dim]")
            
            summary_text = "  |  ".join(summary_parts)
            
            self.console.print()
            self.console.print(
                Panel(
                    summary_text,
                    border_style=status_color,
                    box=box.ROUNDED,
                    padding=(0, 1),
                )
            )
        else:
            status = "Error" if has_error else "Success"
            time_str = f" ({elapsed_time:.2f}s)" if elapsed_time else ""
            print(f"\n{status}{time_str}", file=sys.stderr)


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

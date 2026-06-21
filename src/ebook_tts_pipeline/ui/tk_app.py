from __future__ import annotations

import argparse
import queue
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Callable, Optional, Union

from ebook_tts_pipeline.config import PipelineConfig
from ebook_tts_pipeline.debug_logging import FailureLogger
from ebook_tts_pipeline.ui.controller import ChapterStage, PrototypeUiController


STAGE_COLORS = {
    ChapterStage.RAW: "#d9d9d9",
    ChapterStage.SEGMENTED: "#d9d9d9",
    ChapterStage.ANNOTATED: "#a8d5a2",
    ChapterStage.SCRIPTED: "#8fbbe8",
    ChapterStage.AUDIO: "#f2d36b",
}


class PrototypeTkApp:
    def __init__(self, root: tk.Tk, book_root: str = "books/prototype", fake_tts: bool = False) -> None:
        self.root = root
        self.root.title("Ebook TTS Prototype")
        self.controller = PrototypeUiController(book_root=book_root, fake_tts=fake_tts)
        self.events: queue.Queue = queue.Queue()
        self.registry_visible = tk.BooleanVar(value=False)
        self.epub_path = tk.StringVar()
        self.book_root = tk.StringVar(value=book_root)
        self.book_title = tk.StringVar(value="Untitled Book")
        self.book_slug = tk.StringVar(value=Path(book_root).name or "book")
        self.fake_tts = tk.BooleanVar(value=fake_tts)
        self.status = tk.StringVar(value="Ready")
        self._loading_library = False
        self.registry_fields = {}

        self._build_layout()
        self.refresh()
        self._poll_events()

    def _build_layout(self) -> None:
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)

        top = ttk.Frame(self.root, padding=8)
        top.grid(row=0, column=0, sticky="ew")
        top.columnconfigure(1, weight=1)
        top.columnconfigure(4, weight=1)

        ttk.Label(top, text="EPUB").grid(row=0, column=0, sticky="w")
        ttk.Entry(top, textvariable=self.epub_path).grid(row=0, column=1, sticky="ew", padx=4)
        ttk.Button(top, text="Browse", command=self.choose_epub).grid(row=0, column=2, padx=2)

        ttk.Label(top, text="Book Root").grid(row=0, column=3, sticky="w", padx=(10, 0))
        ttk.Entry(top, textvariable=self.book_root).grid(row=0, column=4, sticky="ew", padx=4)
        ttk.Button(top, text="Folder", command=self.choose_book_root).grid(row=0, column=5, padx=2)

        ttk.Label(top, text="Title").grid(row=1, column=0, sticky="w", pady=(6, 0))
        ttk.Entry(top, textvariable=self.book_title).grid(row=1, column=1, sticky="ew", padx=4, pady=(6, 0))
        ttk.Label(top, text="Slug").grid(row=1, column=3, sticky="w", padx=(10, 0), pady=(6, 0))
        ttk.Entry(top, textvariable=self.book_slug).grid(row=1, column=4, sticky="ew", padx=4, pady=(6, 0))

        actions = ttk.Frame(top)
        actions.grid(row=2, column=0, columnspan=6, sticky="ew", pady=(8, 0))
        ttk.Button(actions, text="Initialize Book", command=self.load_book).pack(side="left")
        ttk.Button(actions, text="Build Global Registry", command=self.build_global_registry).pack(side="left", padx=4)
        ttk.Button(actions, text="Refresh", command=self.refresh).pack(side="left", padx=4)
        ttk.Button(actions, text="Toggle Registry", command=self.toggle_registry).pack(side="left", padx=4)
        ttk.Checkbutton(actions, text="Fake TTS", variable=self.fake_tts).pack(side="left", padx=8)
        ttk.Label(actions, textvariable=self.status).pack(side="right")

        self.main = ttk.Frame(self.root)
        self.main.grid(row=1, column=0, sticky="nsew")
        self.main.columnconfigure(1, weight=1)
        self.main.rowconfigure(0, weight=1)

        library_frame = ttk.Frame(self.main, padding=8)
        library_frame.grid(row=0, column=0, sticky="ns")
        ttk.Label(library_frame, text="Books").pack(anchor="w")
        self.book_list = tk.Listbox(library_frame, width=28, exportselection=False)
        self.book_list.pack(fill="both", expand=True, pady=(4, 0))
        self.book_list.bind("<<ListboxSelect>>", self.select_library_book)
        ttk.Button(library_frame, text="Refresh Books", command=self.load_library).pack(fill="x", pady=(6, 0))

        self.body = ttk.PanedWindow(self.main, orient=tk.HORIZONTAL)
        self.body.grid(row=0, column=1, sticky="nsew")

        self.chapter_frame = ttk.Frame(self.body, padding=8)
        self.body.add(self.chapter_frame, weight=1)
        self.chapter_frame.columnconfigure(0, weight=1)

        self.chapter_canvas = tk.Canvas(self.chapter_frame, highlightthickness=0)
        self.chapter_scrollbar = ttk.Scrollbar(
            self.chapter_frame,
            orient="vertical",
            command=self.chapter_canvas.yview,
        )
        self.chapter_list = ttk.Frame(self.chapter_canvas)
        self.chapter_window = self.chapter_canvas.create_window(
            (0, 0),
            window=self.chapter_list,
            anchor="nw",
        )
        self.chapter_canvas.configure(yscrollcommand=self.chapter_scrollbar.set)
        self.chapter_canvas.grid(row=0, column=0, sticky="nsew")
        self.chapter_scrollbar.grid(row=0, column=1, sticky="ns")
        self.chapter_frame.rowconfigure(0, weight=1)
        self.chapter_list.bind("<Configure>", self._resize_chapter_scroll)
        self.chapter_canvas.bind("<Configure>", self._resize_chapter_width)

        self.registry_frame = ttk.Frame(self.body, padding=8)
        self.registry_frame.columnconfigure(0, weight=1)
        self.registry_frame.rowconfigure(0, weight=1)
        self.registry_canvas = tk.Canvas(self.registry_frame, highlightthickness=0)
        self.registry_scrollbar = ttk.Scrollbar(
            self.registry_frame,
            orient="vertical",
            command=self.registry_canvas.yview,
        )
        self.registry_inner = ttk.Frame(self.registry_canvas)
        self.registry_window = self.registry_canvas.create_window(
            (0, 0),
            window=self.registry_inner,
            anchor="nw",
        )
        self.registry_canvas.configure(yscrollcommand=self.registry_scrollbar.set)
        self.registry_canvas.grid(row=0, column=0, sticky="nsew")
        self.registry_scrollbar.grid(row=0, column=1, sticky="ns")
        self.registry_inner.bind("<Configure>", self._resize_registry_scroll)
        self.registry_canvas.bind("<Configure>", self._resize_registry_width)
        ttk.Button(self.registry_frame, text="Save Registry", command=self.save_registry).grid(
            row=1,
            column=0,
            columnspan=2,
            sticky="e",
            pady=(6, 0),
        )

    def choose_epub(self) -> None:
        selected = filedialog.askopenfilename(
            title="Select EPUB",
            filetypes=[("EPUB files", "*.epub"), ("All files", "*.*")],
        )
        if selected:
            self.epub_path.set(selected)
            title = Path(selected).stem
            self.book_title.set(title)
            self.book_slug.set(_slugify(title))
            self.book_root.set(str(Path("books") / _slugify(title)))

    def choose_book_root(self) -> None:
        selected = filedialog.askdirectory(title="Select Book Output Folder")
        if selected:
            self.book_root.set(selected)

    def load_book(self) -> None:
        epub = self.epub_path.get().strip()
        if not epub:
            messagebox.showerror("Missing EPUB", "Choose an EPUB file first.")
            return

        def work() -> str:
            self._sync_controller()
            result = self.controller.load_epub(
                epub_path=epub,
                title=self.book_title.get().strip() or "Untitled Book",
                slug=self.book_slug.get().strip() or "book",
            )
            return f"Initialized {len(result.chapters)} chapters."

        self._run_background("Initializing book...", work)

    def build_global_registry(self) -> None:
        def work() -> str:
            self._sync_controller()
            count = self.controller.build_global_registry()
            return f"Global registry updated with {count} discovered characters."

        self._run_background("Building global registry...", work)

    def refresh(self) -> None:
        self._sync_controller()
        self.load_library()
        for child in self.chapter_list.winfo_children():
            child.destroy()
        rows = self.controller.chapter_rows()
        if not rows:
            ttk.Label(self.chapter_list, text="No chapters loaded.").grid(row=0, column=0, sticky="w")
        for row_index, row in enumerate(rows):
            button = tk.Button(
                self.chapter_list,
                text=f"{row.index:03d} - {row.title}",
                anchor="w",
                bg=STAGE_COLORS[row.stage],
                relief="raised",
                command=lambda chapter=row.chapter: self.run_chapter_action(chapter),
            )
            button.grid(row=row_index, column=0, sticky="ew", pady=2)
            self.chapter_list.columnconfigure(0, weight=1)
        self.load_registry_panel()

    def toggle_registry(self) -> None:
        if self.registry_visible.get():
            self.body.forget(self.registry_frame)
            self.registry_visible.set(False)
            return
        self.body.add(self.registry_frame, weight=1)
        self.registry_visible.set(True)
        self.load_registry_panel()

    def load_library(self) -> None:
        self._loading_library = True
        try:
            self.book_list.delete(0, "end")
            for index, book in enumerate(self.controller.library_books()):
                self.book_list.insert("end", f"{book.title} ({book.slug})")
                if book.slug == self.controller.current_book_slug:
                    self.book_list.selection_set(index)
        finally:
            self._loading_library = False

    def select_library_book(self, event=None) -> None:
        if self._loading_library:
            return
        selection = self.book_list.curselection()
        if not selection:
            return
        books = self.controller.library_books()
        index = int(selection[0])
        if index >= len(books):
            return
        book = self.controller.select_book(books[index].slug)
        self.book_root.set(str(book.book_root))
        self.book_title.set(book.title)
        self.book_slug.set(book.slug)
        self.epub_path.set(str(book.epub_path))
        self.refresh()

    def load_registry_panel(self) -> None:
        self.registry_fields = {}
        for child in self.registry_inner.winfo_children():
            child.destroy()
        forms = self.controller.registry_character_forms()
        if not forms:
            ttk.Label(self.registry_inner, text="No character registry yet.").grid(row=0, column=0, sticky="w")
            return

        for row_index, form in enumerate(forms):
            card = ttk.LabelFrame(self.registry_inner, text=form.title, padding=8)
            card.grid(row=row_index, column=0, sticky="ew", pady=(0, 8))
            card.columnconfigure(1, weight=1)

            current_row = 0
            for field in form.readonly_fields:
                ttk.Label(card, text=field.label).grid(row=current_row, column=0, sticky="w", pady=1)
                ttk.Label(card, text=field.value, foreground="#555").grid(
                    row=current_row,
                    column=1,
                    sticky="ew",
                    pady=1,
                )
                current_row += 1

            ttk.Separator(card).grid(row=current_row, column=0, columnspan=2, sticky="ew", pady=6)
            current_row += 1
            self.registry_fields[form.role_id] = {}
            for field in form.editable_fields:
                ttk.Label(card, text=field.label).grid(row=current_row, column=0, sticky="nw", pady=2)
                if field.multiline:
                    widget = tk.Text(card, height=4, wrap="word", undo=True)
                    widget.insert("1.0", field.value)
                else:
                    widget = ttk.Entry(card)
                    widget.insert(0, field.value)
                widget.grid(row=current_row, column=1, sticky="ew", pady=2)
                self.registry_fields[form.role_id][field.key] = widget
                current_row += 1

    def save_registry(self) -> None:
        try:
            self._sync_controller()
            for role_id, fields in self.registry_fields.items():
                self.controller.save_registry_character_form(
                    role_id,
                    {key: self._field_value(widget) for key, widget in fields.items()},
                )
        except ValueError as exc:
            messagebox.showerror("Registry Save Error", str(exc))
            return
        self.status.set("Registry saved.")
        self.refresh()

    def run_chapter_action(self, chapter: str) -> None:
        def work() -> str:
            self._sync_controller()
            result = self.controller.run_next_chapter_action(chapter)
            return result.message

        self._run_background(f"Working on {chapter}...", work)

    def _run_background(self, label: str, work: Callable[[], str]) -> None:
        self.status.set(label)
        book_root = self.book_root.get().strip() or "books/prototype"
        log_root = PipelineConfig.from_env(book_root).debug_log_root

        def runner() -> None:
            try:
                self.events.put(("ok", work()))
            except Exception as exc:
                self.events.put(
                    (
                        "error",
                        _pipeline_error_message(
                            exc,
                            label=label,
                            book_root=book_root,
                            log_root=log_root,
                        ),
                    )
                )

        threading.Thread(target=runner, daemon=True).start()

    def _poll_events(self) -> None:
        try:
            while True:
                kind, message = self.events.get_nowait()
                if kind == "ok":
                    self.status.set(message)
                    self.refresh()
                else:
                    self.status.set("Error")
                    messagebox.showerror("Pipeline Error", message)
        except queue.Empty:
            pass
        self.root.after(100, self._poll_events)

    def _sync_controller(self) -> None:
        self.controller.set_book_root(self.book_root.get().strip() or "books/prototype")
        self.controller.fake_tts = bool(self.fake_tts.get())

    def _resize_chapter_scroll(self, event) -> None:
        self.chapter_canvas.configure(scrollregion=self.chapter_canvas.bbox("all"))

    def _resize_chapter_width(self, event) -> None:
        self.chapter_canvas.itemconfigure(self.chapter_window, width=event.width)

    def _resize_registry_scroll(self, event) -> None:
        self.registry_canvas.configure(scrollregion=self.registry_canvas.bbox("all"))

    def _resize_registry_width(self, event) -> None:
        self.registry_canvas.itemconfigure(self.registry_window, width=event.width)

    def _field_value(self, widget) -> str:
        if isinstance(widget, tk.Text):
            return widget.get("1.0", "end").strip()
        return widget.get().strip()


def _slugify(value: str) -> str:
    slug = "".join(ch.lower() if ch.isalnum() else "_" for ch in value).strip("_")
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug or "book"


def _pipeline_error_message(
    exc: BaseException,
    label: str,
    book_root: str,
    log_root: Union[str, Path] = Path("logs") / "annotation_failures",
) -> str:
    log_path = getattr(exc, "debug_log_path", None)
    if not log_path:
        log_path = FailureLogger(
            log_root,
            context={"book_root": book_root, "ui_action": label},
        ).write_failure(
            "ui_pipeline_error",
            {"label": label, "book_root": book_root},
            exc=exc,
        )
    return f"{exc}\n\nDebug log: {log_path}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ebook-tts-ui")
    parser.add_argument("--book-root", default="books/prototype")
    parser.add_argument("--fake-tts", action="store_true")
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    root = tk.Tk()
    PrototypeTkApp(root, book_root=args.book_root, fake_tts=args.fake_tts)
    root.geometry("1100x720")
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

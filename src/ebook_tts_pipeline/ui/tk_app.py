from __future__ import annotations

import argparse
import queue
import time
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Any, Callable, Optional, Union

from ebook_tts_pipeline.config import PipelineConfig
from ebook_tts_pipeline.debug_logging import FailureLogger
from ebook_tts_pipeline.ui.controller import ChapterStage, PrototypeUiController


STAGE_COLORS = {
    ChapterStage.RAW: "#d9d9d9",
    ChapterStage.SEGMENTED: "#d9d9d9",
    ChapterStage.ANNOTATION_REVIEW: "#f0c36d",
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
        self.tts_speed = tk.StringVar(value="1.0")
        self.tts_pause_ms = tk.StringVar(value="250")
        self.tts_intra_pause_ms = tk.StringVar(value="50")
        self.status = tk.StringVar(value="Ready")
        self.read_along_chapter = tk.StringVar()
        self.read_along_selected_unit = tk.IntVar(value=0)
        self.read_along_playback_speed = tk.StringVar(value="1.0")
        self.read_along_generation_mode = tk.StringVar(value="balanced")
        self.read_along_buffer_limit = tk.StringVar(value="2")
        self.read_along_target_buffer_seconds = tk.StringVar(value="20")
        self.read_along_start_buffer_seconds = tk.StringVar(value="20")
        self.read_along_max_buffer_units = tk.StringVar(value="32")
        self.read_along_narrator_voice_type = tk.StringVar(value="male")
        self.read_along_status = tk.StringVar(value="No read-along session.")
        self.read_along_session_active = False
        self.read_along_playing = False
        self.read_along_units = []
        self.current_read_along_session = None
        self.read_along_locked_widgets = []
        self._loading_library = False
        self._loading_read_along_chapters = False
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

        settings = ttk.Frame(top)
        settings.grid(row=3, column=0, columnspan=6, sticky="ew", pady=(6, 0))
        ttk.Label(settings, text="Speed").pack(side="left")
        ttk.Entry(settings, textvariable=self.tts_speed, width=6).pack(side="left", padx=(4, 10))
        ttk.Label(settings, text="Pause ms").pack(side="left")
        ttk.Entry(settings, textvariable=self.tts_pause_ms, width=6).pack(side="left", padx=(4, 10))
        ttk.Label(settings, text="Intra ms").pack(side="left")
        ttk.Entry(settings, textvariable=self.tts_intra_pause_ms, width=6).pack(side="left", padx=(4, 10))
        ttk.Button(settings, text="Save TTS Settings", command=self.save_tts_settings).pack(side="left")

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

        self.tabs = ttk.Notebook(self.main)
        self.tabs.grid(row=0, column=1, sticky="nsew")
        self.audiobook_tab = ttk.Frame(self.tabs)
        self.read_along_tab = ttk.Frame(self.tabs)
        self.tabs.add(self.audiobook_tab, text="Audiobook")
        self.tabs.add(self.read_along_tab, text="Read Along")
        self.audiobook_tab.columnconfigure(0, weight=1)
        self.audiobook_tab.rowconfigure(0, weight=1)

        self.body = ttk.PanedWindow(self.audiobook_tab, orient=tk.HORIZONTAL)
        self.body.grid(row=0, column=0, sticky="nsew")

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
        self._build_read_along_tab()

    def _build_read_along_tab(self) -> None:
        self.read_along_tab.columnconfigure(1, weight=1)
        self.read_along_tab.rowconfigure(0, weight=1)

        sidebar = ttk.Frame(self.read_along_tab, padding=8)
        sidebar.grid(row=0, column=0, sticky="ns")
        sidebar.rowconfigure(1, weight=1)
        ttk.Label(sidebar, text="Contents").grid(row=0, column=0, sticky="w")
        self.read_along_chapter_list = tk.Listbox(sidebar, width=28, exportselection=False)
        self.read_along_chapter_list.grid(row=1, column=0, sticky="ns", pady=(4, 6))
        self.read_along_chapter_list.bind("<<ListboxSelect>>", self.select_read_along_chapter)
        ttk.Button(sidebar, text="Process Book", command=self.process_read_along_book).grid(
            row=2,
            column=0,
            sticky="ew",
            pady=(0, 4),
        )
        ttk.Button(sidebar, text="Build Chapter Units", command=self.rebuild_read_along_chapter).grid(
            row=3,
            column=0,
            sticky="ew",
        )

        reader = ttk.Frame(self.read_along_tab, padding=(0, 8, 8, 8))
        reader.grid(row=0, column=1, sticky="nsew")
        reader.columnconfigure(0, weight=1)
        reader.rowconfigure(1, weight=1)

        controls = ttk.Frame(reader)
        controls.grid(row=0, column=0, sticky="ew", pady=(0, 6))
        controls.columnconfigure(19, weight=1)

        ttk.Label(controls, text="Speed").grid(row=0, column=0, sticky="w")
        speed = ttk.Entry(controls, textvariable=self.read_along_playback_speed, width=6)
        speed.grid(row=0, column=1, padx=(4, 10))
        ttk.Label(controls, text="Generation").grid(row=0, column=2, sticky="w")
        generation = ttk.Combobox(
            controls,
            textvariable=self.read_along_generation_mode,
            values=["balanced", "fast", "precise"],
            width=10,
            state="readonly",
        )
        generation.grid(row=0, column=3, padx=(4, 10))
        ttk.Label(controls, text="Buffer").grid(row=0, column=4, sticky="w")
        buffer_limit = ttk.Spinbox(
            controls,
            from_=1,
            to=8,
            textvariable=self.read_along_buffer_limit,
            width=4,
        )
        buffer_limit.grid(row=0, column=5, padx=(4, 10))
        ttk.Label(controls, text="Buffer s").grid(row=0, column=6, sticky="w")
        target_buffer = ttk.Entry(controls, textvariable=self.read_along_target_buffer_seconds, width=6)
        target_buffer.grid(row=0, column=7, padx=(4, 10))
        ttk.Label(controls, text="Start s").grid(row=0, column=8, sticky="w")
        start_buffer = ttk.Entry(controls, textvariable=self.read_along_start_buffer_seconds, width=6)
        start_buffer.grid(row=0, column=9, padx=(4, 10))
        ttk.Label(controls, text="Max units").grid(row=0, column=10, sticky="w")
        max_units = ttk.Spinbox(
            controls,
            from_=1,
            to=32,
            textvariable=self.read_along_max_buffer_units,
            width=4,
        )
        max_units.grid(row=0, column=11, padx=(4, 10))
        ttk.Label(controls, text="Narrator").grid(row=0, column=12, sticky="w")
        narrator = ttk.Combobox(
            controls,
            textvariable=self.read_along_narrator_voice_type,
            values=["male", "female", "current"],
            width=9,
            state="readonly",
        )
        narrator.grid(row=0, column=13, padx=(4, 10))

        ttk.Button(controls, text="Save", command=self.save_read_along_settings).grid(row=0, column=14, padx=(0, 4))
        self.read_along_start_button = ttk.Button(
            controls,
            text="Start Session",
            command=self.start_read_along_session,
        )
        self.read_along_start_button.grid(row=0, column=15, padx=(0, 4))
        self.read_along_end_button = ttk.Button(
            controls,
            text="End Session",
            command=self.end_read_along_session,
            state="disabled",
        )
        self.read_along_end_button.grid(row=0, column=16, padx=(0, 4))
        ttk.Button(controls, text="Previous Page", command=lambda: self.read_along_text.yview_scroll(-1, "pages")).grid(
            row=0,
            column=17,
            padx=(0, 4),
        )
        ttk.Button(controls, text="Next Page", command=lambda: self.read_along_text.yview_scroll(1, "pages")).grid(
            row=0,
            column=18,
            padx=(0, 4),
        )

        self.read_along_locked_widgets = [
            (self.read_along_chapter_list, "normal"),
            (speed, "normal"),
            (generation, "readonly"),
            (buffer_limit, "normal"),
            (target_buffer, "normal"),
            (start_buffer, "normal"),
            (max_units, "normal"),
            (narrator, "readonly"),
            (self.read_along_start_button, "normal"),
        ]

        text_frame = ttk.Frame(reader)
        text_frame.grid(row=1, column=0, sticky="nsew")
        text_frame.columnconfigure(0, weight=1)
        text_frame.rowconfigure(0, weight=1)
        self.read_along_text = tk.Text(
            text_frame,
            wrap="word",
            padx=42,
            pady=28,
            spacing1=3,
            spacing2=1,
            spacing3=7,
            font=("Georgia", 13),
            undo=False,
            borderwidth=0,
            highlightthickness=1,
            highlightbackground="#cfcfcf",
        )
        self.read_along_text.grid(row=0, column=0, sticky="nsew")
        read_scrollbar = ttk.Scrollbar(text_frame, orient="vertical", command=self.read_along_text.yview)
        read_scrollbar.grid(row=0, column=1, sticky="ns")
        self.read_along_text.configure(yscrollcommand=read_scrollbar.set)
        self.read_along_text.bind("<ButtonRelease-1>", self.select_read_along_unit_at_click)
        self.read_along_text.tag_configure("read_along_quote", foreground="#202020")
        self.read_along_text.tag_configure("read_along_selected", background="#e5edf8")
        self.read_along_text.tag_configure("read_along_current", background="#f7d774")
        self.read_along_text.tag_configure("read_along_buffered", background="#d7ead8")
        self.read_along_text.configure(state="disabled")

        footer = ttk.Frame(reader)
        footer.grid(row=2, column=0, sticky="ew", pady=(6, 0))
        footer.columnconfigure(0, weight=1)
        ttk.Label(footer, textvariable=self.read_along_status).grid(row=0, column=0, sticky="w")

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
        self.load_tts_settings()
        self.load_read_along_settings()
        self.load_library()
        self.load_read_along_chapters()
        for child in self.chapter_list.winfo_children():
            child.destroy()
        rows = self.controller.chapter_rows()
        if not rows:
            ttk.Label(self.chapter_list, text="No chapters loaded.").grid(row=0, column=0, sticky="w")
        for row_index, row in enumerate(rows):
            row_frame = ttk.Frame(self.chapter_list)
            row_frame.grid(row=row_index, column=0, sticky="ew", pady=2)
            row_frame.columnconfigure(0, weight=1)
            button = tk.Button(
                row_frame,
                text=f"{row.index:03d} - {row.title}",
                anchor="w",
                bg=STAGE_COLORS[row.stage],
                relief="raised",
                command=lambda chapter=row.chapter: self.run_chapter_action(chapter),
                state="disabled" if row.stage == ChapterStage.ANNOTATION_REVIEW else "normal",
            )
            button.grid(row=0, column=0, sticky="ew")
            annotation_state = (
                "normal"
                if row.stage
                in {
                    ChapterStage.ANNOTATION_REVIEW,
                    ChapterStage.ANNOTATED,
                    ChapterStage.SCRIPTED,
                    ChapterStage.AUDIO,
                }
                else "disabled"
            )
            ttk.Button(
                row_frame,
                text="Open Annotation",
                command=lambda chapter=row.chapter: self.open_annotation_review(chapter),
                state=annotation_state,
            ).grid(row=0, column=1, padx=(6, 0))
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

    def load_tts_settings(self) -> None:
        settings = self.controller.tts_settings()
        self.tts_speed.set(str(settings["tts_speed"]))
        self.tts_pause_ms.set(str(settings["pause_between_sentences_ms"]))
        self.tts_intra_pause_ms.set(str(settings["intra_sentence_pause_ms"]))

    def save_tts_settings(self) -> None:
        try:
            self._sync_controller()
            self.controller.save_tts_settings(
                {
                    "tts_speed": self.tts_speed.get(),
                    "pause_between_sentences_ms": self.tts_pause_ms.get(),
                    "intra_sentence_pause_ms": self.tts_intra_pause_ms.get(),
                }
            )
        except ValueError as exc:
            messagebox.showerror("TTS Settings Error", str(exc))
            return
        self.status.set("TTS settings saved.")

    def load_read_along_settings(self) -> None:
        if self.read_along_session_active:
            return
        settings = self.controller.read_along_settings()
        self.read_along_playback_speed.set(str(settings["playback_speed"]))
        self.read_along_generation_mode.set(str(settings["generation_mode"]))
        self.read_along_buffer_limit.set(str(settings["buffer_limit"]))
        self.read_along_target_buffer_seconds.set(str(settings["target_buffer_seconds"]))
        self.read_along_start_buffer_seconds.set(str(settings["start_buffer_seconds"]))
        self.read_along_max_buffer_units.set(str(settings["max_buffer_units"]))
        self.read_along_narrator_voice_type.set(str(settings["narrator_voice_type"]))

    def save_read_along_settings(self) -> None:
        try:
            self._sync_controller()
            self.controller.save_read_along_settings(self._read_along_settings_payload())
            settings = self.controller.read_along_settings()
            self.read_along_playback_speed.set(str(settings["playback_speed"]))
            self.read_along_generation_mode.set(str(settings["generation_mode"]))
            self.read_along_buffer_limit.set(str(settings["buffer_limit"]))
            self.read_along_target_buffer_seconds.set(str(settings["target_buffer_seconds"]))
            self.read_along_start_buffer_seconds.set(str(settings["start_buffer_seconds"]))
            self.read_along_max_buffer_units.set(str(settings["max_buffer_units"]))
            self.read_along_narrator_voice_type.set(str(settings["narrator_voice_type"]))
        except ValueError as exc:
            messagebox.showerror("Read-Along Settings Error", str(exc))
            return
        self.read_along_status.set("Read-along settings saved.")

    def load_read_along_chapters(self) -> None:
        if self.read_along_session_active:
            return
        self._loading_read_along_chapters = True
        try:
            self.read_along_chapter_list.delete(0, "end")
            rows = self.controller.chapter_rows()
            selected_index = None
            for index, row in enumerate(rows):
                self.read_along_chapter_list.insert("end", f"{row.index:03d} - {row.title}")
                if row.chapter == self.read_along_chapter.get():
                    selected_index = index
            if rows and selected_index is None and not self.read_along_chapter.get():
                selected_index = 0
            if selected_index is not None:
                self.read_along_chapter_list.selection_clear(0, "end")
                self.read_along_chapter_list.selection_set(selected_index)
                self.read_along_chapter_list.see(selected_index)
                if not self.read_along_chapter.get():
                    self.load_read_along_chapter(rows[selected_index].chapter)
        finally:
            self._loading_read_along_chapters = False

    def select_read_along_chapter(self, event=None) -> None:
        if self._loading_read_along_chapters or self.read_along_session_active:
            return
        selection = self.read_along_chapter_list.curselection()
        if not selection:
            return
        rows = self.controller.chapter_rows()
        index = int(selection[0])
        if index >= len(rows):
            return
        self.load_read_along_chapter(rows[index].chapter)

    def load_read_along_chapter(self, chapter: str) -> None:
        self._sync_controller()
        self.read_along_chapter.set(chapter)
        self.read_along_units = []
        try:
            chapter_text = self.controller.chapter_text(chapter)
        except OSError as exc:
            self._render_read_along_text("")
            self.read_along_status.set(f"Could not load {chapter}: {exc}")
            return

        unit_message = ""
        try:
            self.read_along_units = self.controller.read_along_units(chapter)
        except Exception as exc:
            unit_message = f" Units unavailable: {exc}"

        self._render_read_along_text(chapter_text)
        for unit in self.read_along_units:
            self._tag_read_along_unit(int(unit["unit_id"]), "read_along_quote")

        if self.read_along_units:
            first_unit = int(self.read_along_units[0]["unit_id"])
            self.read_along_selected_unit.set(first_unit)
            self._highlight_read_along_units(selected=first_unit)
            self.read_along_status.set(f"{chapter}: {len(self.read_along_units)} read-along units loaded.")
        else:
            self.read_along_status.set(f"{chapter} loaded.{unit_message}")

    def rebuild_read_along_chapter(self) -> None:
        chapter = self.read_along_chapter.get().strip()
        if not chapter:
            messagebox.showerror("Missing Chapter", "Select a chapter first.")
            return
        if self.read_along_session_active:
            messagebox.showerror("Session Active", "End the active read-along session before rebuilding units.")
            return

        def work() -> dict:
            self._sync_controller()
            units = self.controller.build_read_along_units(chapter)
            return {"chapter": chapter, "count": len(units)}

        def on_ok(result: dict) -> None:
            self.load_read_along_chapter(str(result["chapter"]))
            self.read_along_status.set(f"Built {result['count']} read-along units for {result['chapter']}.")

        self._run_background("Building read-along units...", work, on_ok=on_ok, refresh_after=False)

    def process_read_along_book(self) -> None:
        if self.read_along_session_active:
            messagebox.showerror("Session Active", "End the active read-along session before processing the book.")
            return

        def work() -> dict:
            self._sync_controller()
            return self.controller.process_read_along_book()

        def on_ok(result: dict) -> None:
            self.refresh()
            self.read_along_status.set(
                "Processed read-along book: "
                f"{result['chapters']} chapters, {result['units_built']} unit files."
            )

        self._run_background("Processing read-along book...", work, on_ok=on_ok, refresh_after=False)

    def select_read_along_unit_at_click(self, event=None) -> None:
        if self.read_along_session_active:
            return
        index = self.read_along_text.index("insert")
        count = self.read_along_text.count("1.0", index, "chars")
        if not count:
            return
        offset = int(count[0])
        for unit in self.read_along_units:
            if int(unit["source_start"]) <= offset < int(unit["source_end"]):
                unit_id = int(unit["unit_id"])
                self.read_along_selected_unit.set(unit_id)
                self._highlight_read_along_units(selected=unit_id)
                self.read_along_status.set(
                    f"Selected {unit_id + 1}/{len(self.read_along_units)}: {unit['role']}"
                )
                break

    def start_read_along_session(self) -> None:
        chapter = self.read_along_chapter.get().strip()
        if not chapter:
            messagebox.showerror("Missing Chapter", "Select a chapter first.")
            return
        if self.read_along_session_active:
            return
        try:
            self._sync_controller()
            self.controller.save_read_along_settings(self._read_along_settings_payload())
            settings = self.controller.read_along_settings()
        except ValueError as exc:
            messagebox.showerror("Read-Along Settings Error", str(exc))
            return
        start_unit_id = max(0, int(self.read_along_selected_unit.get()))
        self._set_read_along_controls_locked(True)
        self.read_along_end_button.configure(state="normal")
        self.read_along_status.set("Building initial read-along buffer...")

        def work() -> dict:
            self._sync_controller()
            units = self.controller.read_along_units(chapter)
            session = self.controller.create_read_along_session(chapter, units, settings)
            buffered = session.fill_buffer(start_unit_id=start_unit_id)
            return {
                "chapter": chapter,
                "session": session,
                "units": units,
                "settings": settings,
                "buffered_ids": [item.unit_id for item in buffered],
            }

        def on_ok(result: dict) -> None:
            self.current_read_along_session = result["session"]
            self.read_along_units = list(result["units"])
            self.read_along_session_active = True
            buffered_ids = list(result["buffered_ids"])
            current = buffered_ids[0] if buffered_ids else start_unit_id
            self.read_along_selected_unit.set(current)
            self._highlight_read_along_units(current=current, buffered=buffered_ids[1:])
            self.read_along_status.set(
                f"Read-along session started. Buffer ready: {len(buffered_ids)}/{settings['buffer_limit']}."
            )
            self._play_read_along_current()

        def on_error() -> None:
            self.read_along_session_active = False
            self.current_read_along_session = None
            self.read_along_playing = False
            self._set_read_along_controls_locked(False)
            self.read_along_end_button.configure(state="disabled")

        self._run_background(
            "Building read-along buffer...",
            work,
            on_ok=on_ok,
            on_error=on_error,
            refresh_after=False,
        )

    def end_read_along_session(self, message: str = "Read-along session ended.") -> None:
        session = self.current_read_along_session
        if session is not None:
            session.end()
        self.current_read_along_session = None
        self.read_along_session_active = False
        self.read_along_playing = False
        self._set_read_along_controls_locked(False)
        self.read_along_end_button.configure(state="disabled")
        self._highlight_read_along_units(selected=int(self.read_along_selected_unit.get()))
        self.read_along_status.set(message)

    def _fill_read_along_buffer(self, autoplay: bool) -> None:
        session = self.current_read_along_session
        if not self.read_along_session_active or session is None:
            return

        def work() -> dict:
            buffered = session.fill_buffer()
            return {"buffered_ids": [item.unit_id for item in buffered]}

        def on_ok(result: dict) -> None:
            if not self.read_along_session_active or session is not self.current_read_along_session:
                return
            current = session.peek_ready().unit_id if session.peek_ready() is not None else None
            self._highlight_read_along_units(
                current=current,
                buffered=[unit_id for unit_id in session.ready_unit_ids if unit_id != current],
            )
            if autoplay:
                self._play_read_along_current()

        self._run_background("Filling read-along buffer...", work, on_ok=on_ok, refresh_after=False)

    def _play_read_along_current(self) -> None:
        session = self.current_read_along_session
        if not self.read_along_session_active or self.read_along_playing or session is None:
            return
        item = session.peek_ready()
        if item is None:
            if int(self.read_along_selected_unit.get()) >= len(self.read_along_units) - 1:
                self.end_read_along_session("Reached the end of the chapter.")
            else:
                self._fill_read_along_buffer(autoplay=True)
            return

        unit_id = item.unit_id
        self.read_along_selected_unit.set(unit_id)
        self._highlight_read_along_units(
            current=unit_id,
            buffered=[ready_id for ready_id in session.ready_unit_ids if ready_id != unit_id],
        )
        self.read_along_status.set(f"Playing {unit_id + 1}/{len(self.read_along_units)}.")
        self.read_along_playing = True

        def work() -> dict:
            _play_wav_or_wait(item.audio_path, item.playback_seconds)
            return {"unit_id": unit_id}

        def on_ok(result: dict) -> None:
            self.read_along_playing = False
            if not self.read_along_session_active or session is not self.current_read_along_session:
                return
            session.consume_ready()
            next_unit = int(result["unit_id"]) + 1
            if next_unit < len(self.read_along_units):
                self.read_along_selected_unit.set(next_unit)
                self._fill_read_along_buffer(autoplay=True)
                return
            if session.ready_count:
                self._play_read_along_current()
                return
            self.end_read_along_session("Reached the end of the chapter.")

        self._run_background("Playing read-along audio...", work, on_ok=on_ok, refresh_after=False)

    def _render_read_along_text(self, text: str) -> None:
        self.read_along_text.configure(state="normal")
        self.read_along_text.delete("1.0", "end")
        self.read_along_text.insert("1.0", text)
        for tag in (
            "read_along_quote",
            "read_along_selected",
            "read_along_current",
            "read_along_buffered",
        ):
            self.read_along_text.tag_remove(tag, "1.0", "end")
        self.read_along_text.configure(state="disabled")

    def _highlight_read_along_units(
        self,
        selected: Optional[int] = None,
        current: Optional[int] = None,
        buffered: Optional[list] = None,
    ) -> None:
        self.read_along_text.tag_remove("read_along_selected", "1.0", "end")
        self.read_along_text.tag_remove("read_along_current", "1.0", "end")
        self.read_along_text.tag_remove("read_along_buffered", "1.0", "end")
        if selected is not None:
            self._tag_read_along_unit(selected, "read_along_selected")
            self._see_read_along_unit(selected)
        for unit_id in buffered or []:
            self._tag_read_along_unit(int(unit_id), "read_along_buffered")
        if current is not None:
            self._tag_read_along_unit(current, "read_along_current")
            self._see_read_along_unit(current)

    def _tag_read_along_unit(self, unit_id: int, tag: str) -> None:
        unit = self._read_along_unit(unit_id)
        if unit is None:
            return
        start = int(unit["source_start"])
        end = int(unit["source_end"])
        self.read_along_text.tag_add(tag, f"1.0+{start}c", f"1.0+{end}c")

    def _see_read_along_unit(self, unit_id: int) -> None:
        unit = self._read_along_unit(unit_id)
        if unit is None:
            return
        self.read_along_text.see(f"1.0+{int(unit['source_start'])}c")

    def _read_along_unit(self, unit_id: int) -> Optional[dict]:
        for unit in self.read_along_units:
            if int(unit["unit_id"]) == int(unit_id):
                return unit
        return None

    def _set_read_along_controls_locked(self, locked: bool) -> None:
        for widget, unlocked_state in self.read_along_locked_widgets:
            try:
                widget.configure(state="disabled" if locked else unlocked_state)
            except tk.TclError:
                pass

    def _read_along_settings_payload(self) -> dict:
        target_buffer_text = self.read_along_target_buffer_seconds.get()
        try:
            target_buffer = float(target_buffer_text)
            max_buffer_seconds = str(max(target_buffer, target_buffer * 2))
        except ValueError:
            max_buffer_seconds = target_buffer_text
        return {
            "playback_speed": self.read_along_playback_speed.get(),
            "generation_mode": self.read_along_generation_mode.get(),
            "buffer_limit": self.read_along_buffer_limit.get(),
            "target_buffer_seconds": self.read_along_target_buffer_seconds.get(),
            "start_buffer_seconds": self.read_along_start_buffer_seconds.get(),
            "max_buffer_seconds": max_buffer_seconds,
            "max_buffer_units": self.read_along_max_buffer_units.get(),
            "narrator_voice_type": self.read_along_narrator_voice_type.get(),
        }

    def open_annotation_review(self, chapter: str) -> None:
        self._sync_controller()
        if not self.controller.paths.annotation(chapter).exists():
            messagebox.showerror("Annotation Missing", "Generate annotation for this chapter first.")
            return

        dialog = tk.Toplevel(self.root)
        dialog.title(f"Annotation Review - {chapter}")
        dialog.columnconfigure(0, weight=1)
        body = ttk.Frame(dialog, padding=10)
        body.grid(row=0, column=0, sticky="nsew")
        body.columnconfigure(1, weight=1)

        forms = self.controller.annotation_appearance_forms(chapter)
        selections = {}
        ttk.Label(body, text="Character").grid(row=0, column=0, sticky="w", padx=(0, 8))
        ttk.Label(body, text="Age Stage").grid(row=0, column=1, sticky="w")
        if not forms:
            ttk.Label(body, text="No character roles in this annotation.").grid(
                row=1,
                column=0,
                columnspan=2,
                sticky="w",
                pady=(8, 0),
            )

        for row_index, form in enumerate(forms, start=1):
            ttk.Label(body, text=form.name).grid(row=row_index, column=0, sticky="w", padx=(0, 8), pady=2)
            values = [option.age_stage for option in form.age_stage_options]
            initial = form.current_age_stage if form.current_age_stage in values else values[0]
            selection = tk.StringVar(value=initial)
            combo = ttk.Combobox(
                body,
                textvariable=selection,
                values=values,
                state="readonly",
                width=18,
            )
            combo.grid(row=row_index, column=1, sticky="ew", pady=2)
            selections[form.key] = selection

        actions = ttk.Frame(dialog, padding=(10, 0, 10, 10))
        actions.grid(row=1, column=0, sticky="e")

        def confirm() -> None:
            try:
                self.controller.confirm_annotation_appearances(
                    chapter,
                    {key: variable.get() for key, variable in selections.items()},
                )
            except ValueError as exc:
                messagebox.showerror("Annotation Review Error", str(exc), parent=dialog)
                return
            self.status.set(f"Annotation approved for {chapter}.")
            dialog.destroy()
            self.refresh()

        ttk.Button(actions, text="Confirm Character Appearance", command=confirm).pack(side="right")
        ttk.Button(actions, text="Cancel", command=dialog.destroy).pack(side="right", padx=(0, 6))

    def run_chapter_action(self, chapter: str) -> None:
        def work() -> str:
            self._sync_controller()
            result = self.controller.run_next_chapter_action(chapter)
            return result.message

        self._run_background(f"Working on {chapter}...", work)

    def _run_background(
        self,
        label: str,
        work: Callable[[], Any],
        on_ok: Optional[Callable[[Any], None]] = None,
        refresh_after: bool = True,
        on_error: Optional[Callable[[], None]] = None,
    ) -> None:
        self.status.set(label)
        book_root = self.book_root.get().strip() or "books/prototype"
        log_root = PipelineConfig.from_env(book_root).debug_log_root

        def runner() -> None:
            try:
                self.events.put(("ok", work(), on_ok, refresh_after))
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
                        on_error,
                    )
                )

        threading.Thread(target=runner, daemon=True).start()

    def _poll_events(self) -> None:
        try:
            while True:
                event = self.events.get_nowait()
                kind = event[0]
                if kind == "ok":
                    _, result, on_ok, refresh_after = event
                    if on_ok is None:
                        self.status.set(str(result))
                    else:
                        on_ok(result)
                    if refresh_after:
                        self.refresh()
                else:
                    _, message, on_error = event
                    if on_error is not None:
                        on_error()
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


def _play_wav_or_wait(path: Path, playback_seconds: float) -> None:
    try:
        import winsound

        winsound.PlaySound(str(path), winsound.SND_FILENAME)
    except Exception:
        time.sleep(max(0.0, float(playback_seconds)))


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

from __future__ import annotations

import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from PIL import Image, ImageTk

from kdcjc.core import KDCJCParams, KDCJCResult, encrypt_image, estimate_block_count, suggest_block_size
from kdcjc.io import encrypt_file, load_encrypted_image, preview_encrypted_image


class KDCJCApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("KDCJC - 协同拼图图像加密")
        self.geometry("1180x760")
        self.minsize(980, 640)

        self.original_image: Image.Image | None = None
        self.preview_image: Image.Image | None = None
        self.original_path: Path | None = None
        self.encrypted_path: Path | None = None
        self.restored_path: Path | None = None
        self.photo_refs: list[ImageTk.PhotoImage] = []
        self._worker: threading.Thread | None = None
        self._action_buttons: list[ttk.Button] = []

        self._build_style()
        self._build_layout()

    def _build_style(self) -> None:
        style = ttk.Style(self)
        if "vista" in style.theme_names():
            style.theme_use("vista")
        style.configure("Title.TLabel", font=("Segoe UI", 16, "bold"))
        style.configure("Hint.TLabel", foreground="#666666")

    def _build_layout(self) -> None:
        root = ttk.Frame(self, padding=12)
        root.pack(fill=tk.BOTH, expand=True)

        header = ttk.Frame(root)
        header.pack(fill=tk.X, pady=(0, 10))
        ttk.Label(header, text="KDCJC 图像加密", style="Title.TLabel").pack(anchor=tk.W)
        ttk.Label(
            header,
            text="单文件加密/还原，无需密码",
            style="Hint.TLabel",
        ).pack(anchor=tk.W, pady=(4, 0))

        body = ttk.Frame(root)
        body.pack(fill=tk.BOTH, expand=True)

        self._build_controls(body)
        self._build_preview(body)

    def _build_controls(self, parent: ttk.Frame) -> None:
        panel = ttk.Frame(parent, width=320)
        panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 12))
        panel.pack_propagate(False)

        scroll_wrap = ttk.Frame(panel)
        scroll_wrap.pack(fill=tk.BOTH, expand=True)

        self._controls_canvas = tk.Canvas(scroll_wrap, highlightthickness=0, width=300)
        controls_vbar = ttk.Scrollbar(scroll_wrap, orient=tk.VERTICAL, command=self._controls_canvas.yview)
        self._controls_canvas.configure(yscrollcommand=controls_vbar.set)
        self._controls_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        controls_vbar.pack(side=tk.RIGHT, fill=tk.Y)

        controls = ttk.Frame(self._controls_canvas)
        self._controls_window = self._controls_canvas.create_window((0, 0), window=controls, anchor=tk.NW)

        def _sync_scroll_region(_event=None) -> None:
            self._controls_canvas.configure(scrollregion=self._controls_canvas.bbox("all"))

        def _sync_canvas_width(event: tk.Event) -> None:
            self._controls_canvas.itemconfigure(self._controls_window, width=event.width)

        controls.bind("<Configure>", _sync_scroll_region)
        self._controls_canvas.bind("<Configure>", _sync_canvas_width)

        def _on_mousewheel(event: tk.Event) -> None:
            if event.delta:
                self._controls_canvas.yview_scroll(int(-event.delta / 120), "units")
            elif event.num == 4:
                self._controls_canvas.yview_scroll(-3, "units")
            elif event.num == 5:
                self._controls_canvas.yview_scroll(3, "units")

        for widget in (self._controls_canvas, controls):
            widget.bind("<MouseWheel>", _on_mousewheel)
            widget.bind("<Button-4>", _on_mousewheel)
            widget.bind("<Button-5>", _on_mousewheel)

        file_box = ttk.LabelFrame(controls, text="文件", padding=10)
        file_box.pack(fill=tk.X, pady=(0, 10))
        btn_open = ttk.Button(file_box, text="打开图片", command=self.open_image)
        btn_open.pack(fill=tk.X)
        btn_restore = ttk.Button(file_box, text="打开加密图片并还原", command=self.open_and_restore)
        btn_restore.pack(fill=tk.X, pady=(8, 0))
        self.file_label = ttk.Label(file_box, text="未选择文件", wraplength=280, style="Hint.TLabel")
        self.file_label.pack(anchor=tk.W, pady=(8, 0))

        self.storage_info_var = tk.StringVar(value="还原信息写在像素 LSB 中")
        ttk.Label(controls, textvariable=self.storage_info_var, wraplength=280, style="Hint.TLabel").pack(
            anchor=tk.W, pady=(0, 10)
        )

        param_box = ttk.LabelFrame(controls, text="参数", padding=10)
        param_box.pack(fill=tk.X, pady=(0, 10))
        self.block_size_var = tk.IntVar(value=64)
        self.edge_weight_var = tk.DoubleVar(value=2.0)
        self.holistic_weight_var = tk.DoubleVar(value=0.4)
        self.global_weight_var = tk.DoubleVar(value=0.12)
        self.top_k_var = tk.IntVar(value=3)
        self.swap_iterations_var = tk.IntVar(value=5000)
        self.cluster_method_var = tk.StringVar(value="none")
        self.cluster_piles_var = tk.IntVar(value=12)
        self.pile_layout_var = tk.StringVar(value="voronoi")
        self.soften_strip_var = tk.IntVar(value=0)
        self.pile_toning_var = tk.BooleanVar(value=True)
        self.block_smooth_var = tk.BooleanVar(value=True)

        self._add_spinbox(param_box, "块大小", self.block_size_var, 4, 128, 4)
        self._add_spinbox(param_box, "边缘权重", self.edge_weight_var, 0.1, 5.0, 0.1, is_float=True)
        self._add_spinbox(
            param_box, "整体权重", self.holistic_weight_var, 0.0, 3.0, 0.05, is_float=True
        )
        self._add_spinbox(
            param_box, "全局权重", self.global_weight_var, 0.0, 3.0, 0.05, is_float=True
        )
        self._add_spinbox(param_box, "Top-K 随机", self.top_k_var, 1, 20, 1)
        self._add_spinbox(param_box, "交换优化", self.swap_iterations_var, 0, 10000, 500)
        cluster_row = ttk.Frame(param_box)
        cluster_row.pack(fill=tk.X, pady=4)
        ttk.Label(cluster_row, text="分堆方法").pack(side=tk.LEFT)
        cluster_combo = ttk.Combobox(
            cluster_row,
            textvariable=self.cluster_method_var,
            values=("pca", "umap", "none"),
            state="readonly",
            width=8,
        )
        cluster_combo.pack(side=tk.RIGHT)
        self._add_spinbox(param_box, "堆数", self.cluster_piles_var, 2, 64, 1)
        layout_row = ttk.Frame(param_box)
        layout_row.pack(fill=tk.X, pady=4)
        ttk.Label(layout_row, text="放置方式").pack(side=tk.LEFT)
        layout_combo = ttk.Combobox(
            layout_row,
            textvariable=self.pile_layout_var,
            values=("heap", "strip", "voronoi"),
            state="readonly",
            width=8,
        )
        layout_combo.pack(side=tk.RIGHT)
        self._add_spinbox(param_box, "接缝柔化带", self.soften_strip_var, 0, 3, 1)
        toning_row = ttk.Frame(param_box)
        toning_row.pack(fill=tk.X, pady=4)
        ttk.Label(toning_row, text="分堆色调协调").pack(side=tk.LEFT)
        ttk.Checkbutton(toning_row, variable=self.pile_toning_var).pack(side=tk.RIGHT)
        smooth_row = ttk.Frame(param_box)
        smooth_row.pack(fill=tk.X, pady=4)
        ttk.Label(smooth_row, text="块内均值平滑").pack(side=tk.LEFT)
        ttk.Checkbutton(smooth_row, variable=self.block_smooth_var).pack(side=tk.RIGHT)
        self.block_count_label = ttk.Label(param_box, text="预计块数：-", style="Hint.TLabel")
        self.block_count_label.pack(anchor=tk.W, pady=(6, 0))
        self.block_size_var.trace_add("write", lambda *_: self._update_block_estimate())

        action_box = ttk.LabelFrame(controls, text="操作", padding=10)
        action_box.pack(fill=tk.X, pady=(0, 10))
        btn_save = ttk.Button(action_box, text="加密并保存图片", command=self.encrypt_and_save)
        btn_save.pack(fill=tk.X)
        btn_preview = ttk.Button(action_box, text="预览加密效果", command=self.preview_encrypt)
        btn_preview.pack(fill=tk.X, pady=(8, 0))
        self._action_buttons = [btn_open, btn_restore, btn_save, btn_preview]

        self.progress = ttk.Progressbar(controls, mode="determinate", maximum=100)
        self.progress.pack(fill=tk.X, pady=(0, 10))

        info_box = ttk.LabelFrame(controls, text="说明", padding=10)
        info_box.pack(fill=tk.X, pady=(0, 10))
        ttk.Label(
            info_box,
            text=(
                "加密：打开原图 → 加密并保存 PNG\n"
                "还原：打开加密图 → 保存还原图\n"
                "请用 PNG，勿经微信/压缩重存"
            ),
            wraplength=280,
            justify=tk.LEFT,
        ).pack(anchor=tk.W)

        self.status_var = tk.StringVar(value="就绪")
        ttk.Label(controls, textvariable=self.status_var, style="Hint.TLabel", wraplength=280).pack(
            anchor=tk.W, pady=(0, 8)
        )

    def _build_preview(self, parent: ttk.Frame) -> None:
        preview_box = ttk.LabelFrame(parent, text="预览", padding=10)
        preview_box.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        toolbar = ttk.Frame(preview_box)
        toolbar.pack(fill=tk.X, pady=(0, 8))
        self.view_mode = tk.StringVar(value="split")
        ttk.Radiobutton(
            toolbar, text="左右对比", value="split", variable=self.view_mode, command=self.refresh_preview
        ).pack(side=tk.LEFT)
        ttk.Radiobutton(
            toolbar, text="仅原图", value="original", variable=self.view_mode, command=self.refresh_preview
        ).pack(side=tk.LEFT, padx=(10, 0))
        ttk.Radiobutton(
            toolbar, text="仅结果", value="result", variable=self.view_mode, command=self.refresh_preview
        ).pack(side=tk.LEFT, padx=(10, 0))

        canvas_wrap = ttk.Frame(preview_box)
        canvas_wrap.pack(fill=tk.BOTH, expand=True)
        self.canvas = tk.Canvas(canvas_wrap, background="#1e1e1e", highlightthickness=0)
        vbar = ttk.Scrollbar(canvas_wrap, orient=tk.VERTICAL, command=self.canvas.yview)
        hbar = ttk.Scrollbar(canvas_wrap, orient=tk.HORIZONTAL, command=self.canvas.xview)
        self.canvas.configure(yscrollcommand=vbar.set, xscrollcommand=hbar.set)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        vbar.grid(row=0, column=1, sticky="ns")
        hbar.grid(row=1, column=0, sticky="ew")
        canvas_wrap.rowconfigure(0, weight=1)
        canvas_wrap.columnconfigure(0, weight=1)
        self.canvas.bind("<Configure>", lambda _event: self.refresh_preview())

    def _add_spinbox(
        self,
        parent: ttk.Frame,
        label: str,
        variable: tk.Variable,
        from_: float,
        to: float,
        increment: float,
        is_float: bool = False,
    ) -> None:
        row = ttk.Frame(parent)
        row.pack(fill=tk.X, pady=4)
        ttk.Label(row, text=label, width=10).pack(side=tk.LEFT)
        if is_float:
            widget = ttk.Spinbox(
                row,
                textvariable=variable,
                from_=from_,
                to=to,
                increment=increment,
                width=10,
            )
        else:
            widget = ttk.Spinbox(
                row,
                textvariable=variable,
                from_=int(from_),
                to=int(to),
                increment=int(increment),
                width=10,
            )
        widget.pack(side=tk.RIGHT)

    def _params(self) -> KDCJCParams:
        return KDCJCParams(
            block_size=int(self.block_size_var.get()),
            edge_weight=float(self.edge_weight_var.get()),
            holistic_weight=float(self.holistic_weight_var.get()),
            global_weight=float(self.global_weight_var.get()),
            top_k=int(self.top_k_var.get()),
            swap_iterations=int(self.swap_iterations_var.get()),
            cluster_method=str(self.cluster_method_var.get()).strip().lower(),
            cluster_piles=int(self.cluster_piles_var.get()),
            pile_layout=str(self.pile_layout_var.get()).strip().lower(),
            soften_strip=int(self.soften_strip_var.get()),
            pile_toning=bool(self.pile_toning_var.get()),
            block_smooth=bool(self.block_smooth_var.get()),
        )

    def _update_block_estimate(self) -> None:
        if self.original_image is None:
            self.block_count_label.configure(text="预计块数：-")
            return
        try:
            block_size = int(self.block_size_var.get())
            if block_size < 4:
                self.block_count_label.configure(text="预计块数：-（块大小至少 4）")
                return
            gw, gh, count = estimate_block_count(
                self.original_image.size[0], self.original_image.size[1], block_size
            )
            hint = f"预计块数：{count}（{gw}×{gh}）"
            if count > 12000:
                hint += "，过大！"
            elif count > 4000:
                hint += "，可能较慢"
            self.block_count_label.configure(text=hint)
        except (tk.TclError, ValueError, ZeroDivisionError):
            self.block_count_label.configure(text="预计块数：-")

    def _set_busy(self, busy: bool) -> None:
        state = tk.DISABLED if busy else tk.NORMAL
        for button in self._action_buttons:
            button.configure(state=state)

    def _set_progress(self, value: int, text: str) -> None:
        self.progress["value"] = value
        self.status_var.set(text)

    def _run_in_background(self, task_name: str, worker, on_success, on_error) -> None:
        if self._worker and self._worker.is_alive():
            messagebox.showwarning("提示", "当前已有任务在运行，请稍候")
            return

        self._set_busy(True)
        self._set_progress(0, f"正在{task_name}...")

        def progress_callback(done: int, total: int) -> None:
            percent = int(done * 100 / max(total, 1))
            self.after(0, lambda p=percent: self._set_progress(p, f"正在{task_name}... {p}%"))

        def run() -> None:
            try:
                result = worker(progress_callback)
                self.after(0, lambda r=result: on_success(r))
            except Exception as exc:
                err = exc
                self.after(0, lambda e=err: on_error(e))
            finally:
                self.after(0, self._finish_worker)

        self._worker = threading.Thread(target=run, daemon=True)
        self._worker.start()

    def _finish_worker(self) -> None:
        self._set_busy(False)

    def open_image(self) -> None:
        path = filedialog.askopenfilename(
            title="选择图片",
            filetypes=[
                ("Image Files", "*.png;*.jpg;*.jpeg;*.bmp;*.webp"),
                ("All Files", "*.*"),
            ],
        )
        if not path:
            return
        try:
            self.original_image = Image.open(path).convert("RGB")
            self.preview_image = None
            self.original_path = Path(path)
            self.encrypted_path = None
            self.storage_info_var.set("还原数据：加密后写入像素")
            self.file_label.configure(text=f"图片: {self.original_path.name}")

            w, h = self.original_image.size
            suggested = suggest_block_size(w, h)
            _, _, count_default = estimate_block_count(w, h, int(self.block_size_var.get()))
            if count_default > 4000:
                self.block_size_var.set(suggested)

            self._update_block_estimate()
            self.status_var.set(f"已加载图片 {w}×{h}")
            self.refresh_preview()
        except OSError as exc:
            messagebox.showerror("错误", f"无法打开图片: {exc}")

    def _prompt_save_restored(self, restored: Image.Image) -> Path | None:
        if self.encrypted_path is None:
            return None
        default_name = self.encrypted_path.stem + "_restored.png"
        output_path = filedialog.asksaveasfilename(
            title="保存还原图片",
            defaultextension=".png",
            initialdir=str(self.encrypted_path.parent),
            initialfile=default_name,
            filetypes=[
                ("PNG Image", "*.png"),
                ("JPEG Image", "*.jpg;*.jpeg"),
            ],
        )
        if not output_path:
            return None
        path = Path(output_path)
        suffix = path.suffix.lower()
        if suffix in (".jpg", ".jpeg"):
            restored.convert("RGB").save(path, format="JPEG", quality=95)
        else:
            restored.convert("RGB").save(path, format="PNG")
        return path

    def open_and_restore(self) -> None:
        path = filedialog.askopenfilename(
            title="选择加密图片",
            filetypes=[
                ("Encrypted Images", "*.png;*.jpg;*.jpeg;*.kdcjc"),
                ("All Files", "*.*"),
            ],
        )
        if not path:
            return
        self.encrypted_path = Path(path)

        def worker(progress_callback):
            return load_encrypted_image(self.encrypted_path, progress_callback=progress_callback)

        def on_success(result):
            stored, restored, meta = result
            self.original_image = restored
            self.preview_image = stored
            self.view_mode.set("split")
            self.storage_info_var.set("还原数据：已从像素读取")
            self._set_progress(100, "正在还原... 100%")
            saved_path = self._prompt_save_restored(restored)
            if saved_path is None:
                self.restored_path = None
                self.file_label.configure(text=f"已还原（未保存）: {self.encrypted_path.name}")
                self.status_var.set(
                    f"还原完成但未保存：{meta['grid_width']}×{meta['grid_height']} 块"
                )
            else:
                self.restored_path = saved_path
                self.file_label.configure(text=f"已保存: {saved_path.name}")
                self.status_var.set(
                    f"还原已保存：块大小 {meta['block_size']}，网格 {meta['grid_width']}×{meta['grid_height']}"
                )
                messagebox.showinfo("完成", f"已保存:\n{saved_path}")
            self.refresh_preview()

        def on_error(exc):
            messagebox.showerror("错误", str(exc))

        self._run_in_background("还原", worker, on_success, on_error)

    def preview_encrypt(self) -> None:
        if self.original_image is None:
            messagebox.showwarning("提示", "请先打开图片")
            return

        image = self.original_image.copy()
        params = self._params()
        orig_w, orig_h = image.size

        def worker(progress_callback):
            result = encrypt_image(image, params, progress_callback=progress_callback)
            output = preview_encrypted_image(result, orig_w, orig_h)
            result.image = output
            return result

        def on_success(result: KDCJCResult):
            self.preview_image = result.image
            self.storage_info_var.set("还原数据：已写入像素（预览）")
            self.status_var.set(
                f"预览完成：{result.grid_width}×{result.grid_height} 块，边缘已处理"
            )
            self.refresh_preview()

        def on_error(exc):
            messagebox.showerror("错误", str(exc))

        self._run_in_background("预览加密", worker, on_success, on_error)

    def encrypt_and_save(self) -> None:
        if self.original_image is None or self.original_path is None:
            messagebox.showwarning("提示", "请先打开图片")
            return

        default_name = self.original_path.stem + "_encrypted.png"
        output_path = filedialog.asksaveasfilename(
            title="保存加密图片",
            defaultextension=".png",
            initialfile=default_name,
            filetypes=[
                ("PNG Image", "*.png"),
                ("KDCJC Archive", "*.kdcjc"),
            ],
        )
        if not output_path:
            return

        params = self._params()
        input_path = self.original_path

        def worker(progress_callback):
            return encrypt_file(input_path, output_path, params, progress_callback=progress_callback)

        def on_success(result: KDCJCResult):
            self.preview_image = result.image
            self.encrypted_path = Path(output_path)
            self.storage_info_var.set("还原数据：已写入像素")
            self.file_label.configure(text=f"已保存: {Path(output_path).name}")
            self.status_var.set(f"加密完成，块网格 {result.grid_width}×{result.grid_height}")
            self.refresh_preview()
            messagebox.showinfo("完成", f"已保存:\n{output_path}")

        def on_error(exc):
            messagebox.showerror("错误", str(exc))

        self._run_in_background("加密", worker, on_success, on_error)

    def refresh_preview(self) -> None:
        self.canvas.delete("all")
        self.photo_refs.clear()

        left = self.original_image
        right = self.preview_image
        mode = self.view_mode.get()

        if mode == "original":
            right = None
        elif mode == "result":
            left = None

        if left is None and right is None:
            self.canvas.create_text(
                20,
                20,
                anchor=tk.NW,
                fill="#dddddd",
                text="打开图片或执行加密/还原后在此预览",
            )
            self.canvas.configure(scrollregion=(0, 0, 800, 500))
            return

        canvas_w = max(self.canvas.winfo_width(), 400)
        canvas_h = max(self.canvas.winfo_height(), 400)
        gap = 16 if left and right else 0
        slot_w = (canvas_w - gap - 24) // (2 if left and right else 1)

        x_cursor = 12
        y_base = 12
        max_h = 0
        total_w = 12

        if left is not None:
            photo, w, h = self._make_photo(left, slot_w, canvas_h - 24)
            self.photo_refs.append(photo)
            self.canvas.create_text(x_cursor, y_base, anchor=tk.NW, fill="#cccccc", text="原图 / 还原结果")
            self.canvas.create_image(x_cursor, y_base + 18, anchor=tk.NW, image=photo)
            x_cursor += w + gap
            max_h = max(max_h, h + 18)
            total_w = x_cursor

        if right is not None:
            photo, w, h = self._make_photo(right, slot_w, canvas_h - 24)
            self.photo_refs.append(photo)
            self.canvas.create_text(x_cursor, y_base, anchor=tk.NW, fill="#cccccc", text="加密图")
            self.canvas.create_image(x_cursor, y_base + 18, anchor=tk.NW, image=photo)
            max_h = max(max_h, h + 18)
            total_w = x_cursor + w + 12

        self.canvas.configure(scrollregion=(0, 0, max(total_w, canvas_w), max(max_h + 24, canvas_h)))

    def _make_photo(
        self, image: Image.Image, max_w: int, max_h: int
    ) -> tuple[ImageTk.PhotoImage, int, int]:
        scale = min(max_w / image.width, max_h / image.height, 1.0)
        display_size = (max(1, int(image.width * scale)), max(1, int(image.height * scale)))
        resized = image.resize(display_size, Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(resized)
        return photo, display_size[0], display_size[1]


def main() -> None:
    app = KDCJCApp()
    app.mainloop()


if __name__ == "__main__":
    main()

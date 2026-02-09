import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
from sklearn.cluster import KMeans
import threading


class VectorQuantizerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Vector Quantizer - Image Compression")
        self.root.geometry("1200x700")

        # Dark theme colors
        self.bg_dark = "#1e1e1e"
        self.bg_secondary = "#2d2d2d"
        self.bg_hover = "#3e3e3e"
        self.fg_primary = "#ffffff"
        self.fg_secondary = "#b0b0b0"
        self.accent = "#007acc"
        self.accent_hover = "#005a9e"
        self.success = "#4ec9b0"
        self.error = "#f48771"

        self.root.configure(bg=self.bg_dark)

        # Variables
        self.image_path = None
        self.original_image = None
        self.compressed_data_file = None
        self.preview_image = None
        self.compressed_preview = None
        self.decompressed_preview = None

        self.setup_ui()

    def setup_ui(self):
        # Title
        title_frame = tk.Frame(self.root, bg=self.bg_dark)
        title_frame.pack(pady=20)

        title = tk.Label(
            title_frame,
            text="Vector Quantizer",
            font=("Segoe UI", 24, "bold"),
            bg=self.bg_dark,
            fg=self.fg_primary
        )
        title.pack()

        # Notebook for tabs
        style = ttk.Style()
        style.theme_use('default')

        style.configure(
            "TNotebook",
            background=self.bg_dark,
        )
        style.configure(
            "TNotebook.Tab",
            background=self.bg_secondary,
            foreground=self.fg_primary,
            padding=[20, 10],
            font=("Segoe UI", 15)
        )
        style.map(
            "TNotebook.Tab",
            background=[("selected", self.accent)],
            foreground=[("selected", self.fg_primary)]
        )

        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill="both", expand=True, padx=20, pady=10)

        # Compression Tab
        self.compress_frame = tk.Frame(self.notebook, bg=self.bg_dark)
        self.notebook.add(self.compress_frame, text="Compress")
        self.setup_compress_tab()

        # Decompression Tab
        self.decompress_frame = tk.Frame(self.notebook, bg=self.bg_dark)
        self.notebook.add(self.decompress_frame, text="Decompress")
        self.setup_decompress_tab()

    def setup_compress_tab(self):
        # Main container with left and right panels
        main_container = tk.Frame(self.compress_frame, bg=self.bg_dark)
        main_container.pack(fill="both", expand=True, padx=10, pady=10)

        # Left panel
        left_panel = tk.Frame(main_container, bg=self.bg_dark)
        left_panel.pack(side="left", fill="both", expand=True, padx=(0, 5))

        # Image selection
        select_frame = tk.Frame(left_panel, bg=self.bg_dark)
        select_frame.pack(pady=10, padx=10, fill="x")

        self.select_btn = self.create_button(
            select_frame,
            "Select Image",
            self.select_image,
            width=20
        )
        self.select_btn.pack(side="left")

        self.image_label = tk.Label(
            select_frame,
            text="No image selected",
            font=("Segoe UI", 15),
            bg=self.bg_dark,
            fg=self.fg_secondary
        )
        self.image_label.pack(side="left", padx=20)

        # Parameters frame
        params_frame = tk.Frame(left_panel, bg=self.bg_secondary)
        params_frame.pack(pady=10, padx=10, fill="x")

        params_title = tk.Label(
            params_frame,
            text="Compression Parameters",
            font=("Segoe UI", 12, "bold"),
            bg=self.bg_secondary,
            fg=self.fg_primary
        )
        params_title.pack(pady=10)

        # Block height
        self.create_param_row(params_frame, "Block Height:", "8")
        self.block_height_entry = self.last_entry

        # Block width
        self.create_param_row(params_frame, "Block Width:", "8")
        self.block_width_entry = self.last_entry

        # Bits
        self.create_param_row(params_frame, "Bits:", "3")
        self.bits_entry = self.last_entry

        # Output file
        self.create_param_row(params_frame, "Output File:", "compressed_data.npz")
        self.output_entry = self.last_entry

        # Compress button
        compress_btn_frame = tk.Frame(left_panel, bg=self.bg_dark)
        compress_btn_frame.pack(pady=15)

        self.compress_btn = self.create_button(
            compress_btn_frame,
            "Compress Image",
            self.compress_image,
            width=30
        )
        self.compress_btn.pack()

        # Status area
        status_frame = tk.Frame(left_panel, bg=self.bg_secondary)
        status_frame.pack(pady=10, padx=10, fill="both", expand=True)

        status_title = tk.Label(
            status_frame,
            text="Status",
            font=("Segoe UI", 15, "bold"),
            bg=self.bg_secondary,
            fg=self.fg_primary
        )
        status_title.pack(pady=5)

        self.compress_status = tk.Text(
            status_frame,
            height=8,
            font=("Consolas", 15),
            bg=self.bg_dark,
            fg=self.fg_secondary,
            relief="flat",
            padx=10,
            pady=10
        )
        self.compress_status.pack(pady=5, padx=10, fill="both", expand=True)

        # Right panel -> Preview
        right_panel = tk.Frame(main_container, bg=self.bg_secondary, width=400)
        right_panel.pack(side="right", fill="both", padx=(5, 0))
        right_panel.pack_propagate(False)

        preview_title = tk.Label(
            right_panel,
            text="Image Preview",
            font=("Segoe UI", 12, "bold"),
            bg=self.bg_secondary,
            fg=self.fg_primary
        )
        preview_title.pack(pady=10)

        # Original preview
        orig_label = tk.Label(
            right_panel,
            text="Original",
            font=("Segoe UI", 9),
            bg=self.bg_secondary,
            fg=self.fg_secondary
        )
        orig_label.pack(pady=(5, 2))

        self.compress_original_preview = tk.Label(
            right_panel,
            text="No image loaded",
            bg=self.bg_dark,
            fg=self.fg_secondary,
            font=("Segoe UI", 10)
        )
        self.compress_original_preview.pack(pady=5, padx=10)



    def setup_decompress_tab(self):
        # Main container
        main_container = tk.Frame(self.decompress_frame, bg=self.bg_dark)
        main_container.pack(fill="both", expand=True, padx=10, pady=10)

        # Left panel
        left_panel = tk.Frame(main_container, bg=self.bg_dark)
        left_panel.pack(side="left", fill="both", expand=True, padx=(0, 5))

        # File selection
        select_frame = tk.Frame(left_panel, bg=self.bg_dark)
        select_frame.pack(pady=20, padx=20, fill="x")

        self.select_compressed_btn = self.create_button(
            select_frame,
            "Select Compressed File",
            self.select_compressed_file,
            width=25
        )
        self.select_compressed_btn.pack(side="left")

        self.compressed_label = tk.Label(
            select_frame,
            text="No file selected",
            font=("Segoe UI", 15),
            bg=self.bg_dark,
            fg=self.fg_secondary
        )
        self.compressed_label.pack(side="left", padx=20)

        # Output name
        output_frame = tk.Frame(left_panel, bg=self.bg_secondary)
        output_frame.pack(pady=10, padx=20, fill="x")

        self.create_param_row(output_frame, "Output Image Name:", "decompressed_img.png")
        self.decompress_output_entry = self.last_entry

        # Decompress button
        decompress_btn_frame = tk.Frame(left_panel, bg=self.bg_dark)
        decompress_btn_frame.pack(pady=20)

        self.decompress_btn = self.create_button(
            decompress_btn_frame,
            "Decompress Image",
            self.decompress_image,
            width=30
        )
        self.decompress_btn.pack()

        # Status area
        status_frame = tk.Frame(left_panel, bg=self.bg_secondary)
        status_frame.pack(pady=10, padx=20, fill="both", expand=True)

        status_title = tk.Label(
            status_frame,
            text="Status",
            font=("Segoe UI", 15, "bold"),
            bg=self.bg_secondary,
            fg=self.fg_primary
        )
        status_title.pack(pady=5)

        self.decompress_status = tk.Text(
            status_frame,
            height=10,
            font=("Consolas", 15),
            bg=self.bg_dark,
            fg=self.fg_secondary,
            relief="flat",
            padx=10,
            pady=10
        )
        self.decompress_status.pack(pady=5, padx=10, fill="both", expand=True)

        # Right panel -> Preview
        right_panel = tk.Frame(main_container, bg=self.bg_secondary, width=400)
        right_panel.pack(side="right", fill="both", padx=(5, 0))
        right_panel.pack_propagate(False)

        preview_title = tk.Label(
            right_panel,
            text="Decompressed Preview",
            font=("Segoe UI", 12, "bold"),
            bg=self.bg_secondary,
            fg=self.fg_primary
        )
        preview_title.pack(pady=10)

        self.decompress_preview = tk.Label(
            right_panel,
            text="Decompress to see result",
            bg=self.bg_dark,
            fg=self.fg_secondary,
            font=("Segoe UI", 10)
        )
        self.decompress_preview.pack(pady=20, padx=10, expand=True)

    def create_button(self, parent, text, command, width=15):
        btn = tk.Button(
            parent,
            text=text,
            command=command,
            font=("Segoe UI", 10),
            bg=self.accent,
            fg=self.fg_primary,
            activebackground=self.accent_hover,
            activeforeground=self.fg_primary,
            relief="flat",
            padx=20,
            pady=10,
            cursor="hand2",
            width=width
        )

        btn.bind("<Enter>", lambda e: btn.config(bg=self.accent_hover))
        btn.bind("<Leave>", lambda e: btn.config(bg=self.accent))

        return btn

    def create_param_row(self, parent, label_text, default_value):
        row = tk.Frame(parent, bg=self.bg_secondary)
        row.pack(pady=8, padx=20, fill="x")

        label = tk.Label(
            row,
            text=label_text,
            font=("Segoe UI", 10),
            bg=self.bg_secondary,
            fg=self.fg_primary,
            width=20,
            anchor="w"
        )
        label.pack(side="left")

        entry = tk.Entry(
            row,
            font=("Segoe UI", 10),
            bg=self.bg_dark,
            fg=self.fg_primary,
            relief="flat",
            insertbackground=self.fg_primary
        )
        entry.insert(0, default_value)
        entry.pack(side="left", fill="x", expand=True, ipady=5, padx=(10, 0))

        self.last_entry = entry

    def select_image(self):
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[
                ("Image files", "*.png *.jpg *.jpeg *.bmp *.gif"),
                ("All files", "*.*")
            ]
        )
        if file_path:
            self.image_path = file_path
            self.image_label.config(text=file_path.split("/")[-1], fg=self.success)

            # Load and display preview
            try:
                img = Image.open(file_path)
                self.display_preview(img, self.compress_original_preview, 350)
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load image: {str(e)}")

    def select_compressed_file(self):
        file_path = filedialog.askopenfilename(
            title="Select Compressed File",
            filetypes=[("NPZ files", "*.npz"), ("All files", "*.*")]
        )
        if file_path:
            self.compressed_data_file = file_path
            self.compressed_label.config(text=file_path.split("/")[-1], fg=self.success)

    def update_status(self, text_widget, message, color=None):
        text_widget.insert("end", message + "\n")
        if color:
            text_widget.tag_add("color", "end-2l", "end-1l")
            text_widget.tag_config("color", foreground=color)
        text_widget.see("end")
        self.root.update()

    def display_preview(self, img, label_widget, max_size=350):
        img_copy = img.copy()
        img_copy.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(img_copy)
        label_widget.config(image=photo, text="")
        label_widget.image = photo

    def compress_image(self):
        if not self.image_path:
            messagebox.showerror("Error", "Please select an image first!")
            return

        try:
            block_height = int(self.block_height_entry.get())
            block_width = int(self.block_width_entry.get())
            bits = int(self.bits_entry.get())
            output_file = self.output_entry.get()
        except ValueError:
            messagebox.showerror("Error", "Please enter valid numeric values!")
            return

        self.compress_status.delete("1.0", "end")
        self.compress_btn.config(state="disabled")

        def compress_thread():
            try:
                self.update_status(self.compress_status, "Loading image...", self.fg_primary)
                image = Image.open(self.image_path)
                image_array = np.array(image)

                self.update_status(self.compress_status, f"Image shape: {image_array.shape}", self.fg_secondary)

                if len(image_array.shape) == 2:
                    image_array = np.expand_dims(image_array, axis=2)

                height, width, channels = image_array.shape

                pad_height = (block_height - (height % block_height)) % block_height
                pad_width = (block_width - (width % block_width)) % block_width

                padded_image = np.pad(image_array, ((0, pad_height), (0, pad_width), (0, 0)), mode='edge')

                self.update_status(self.compress_status, "Creating blocks...", self.fg_primary)
                blocks = []
                for i in range(0, padded_image.shape[0], block_height):
                    for j in range(0, padded_image.shape[1], block_width):
                        block = padded_image[i:i + block_height, j:j + block_width, :]
                        blocks.append(block.flatten())

                block_vectors = np.array(blocks)
                self.update_status(self.compress_status, f"Total blocks: {block_vectors.shape[0]}", self.fg_secondary)

                num_clusters = 2 ** bits
                self.update_status(self.compress_status, f"Applying K-Means with {num_clusters} clusters(=2^bits)...",
                                   self.fg_primary)

                kmeans = KMeans(n_clusters=num_clusters, random_state=42)
                kmeans.fit(block_vectors)
                codebook = kmeans.cluster_centers_
                labels = kmeans.labels_

                uncompressed_bytes = padded_image.size * padded_image.itemsize
                compressed_bytes = labels.size * labels.itemsize + codebook.size * codebook.itemsize
                ratio = uncompressed_bytes / compressed_bytes

                self.update_status(self.compress_status, f"Compression ratio: {ratio:.2f}", self.success)

                np.savez(output_file,
                         codebook=codebook,
                         labels=labels,
                         original_height=height,
                         original_width=width,
                         block_height=block_height,
                         block_width=block_width,
                         pad_height=pad_height,
                         pad_width=pad_width,
                         channels=channels)

                self.update_status(self.compress_status, f">>> Compression complete! Saved as '{output_file}'",
                                   self.success)

            except Exception as e:
                self.update_status(self.compress_status, f"✗ Error: {str(e)}", self.error)
            finally:
                self.compress_btn.config(state="normal")

        threading.Thread(target=compress_thread, daemon=True).start()

    def decompress_image(self):
        if not self.compressed_data_file:
            messagebox.showerror("Error", "Please select a compressed file first!")
            return

        output_name = self.decompress_output_entry.get()

        self.decompress_status.delete("1.0", "end")
        self.decompress_btn.config(state="disabled")

        def decompress_thread():
            try:
                self.update_status(self.decompress_status, "Loading compressed data...", self.fg_primary)
                data = np.load(self.compressed_data_file)

                codebook = data["codebook"]
                labels = data["labels"]
                original_height = int(data["original_height"])
                original_width = int(data["original_width"])
                block_height = int(data["block_height"])
                block_width = int(data["block_width"])
                pad_height = int(data["pad_height"])
                pad_width = int(data["pad_width"])
                channels = int(data["channels"])

                self.update_status(self.decompress_status, f"Original dimensions: {original_height}x{original_width}",
                                   self.fg_secondary)
                self.update_status(self.decompress_status, "Reconstructing image...", self.fg_primary)

                padded_height = original_height + pad_height
                padded_width = original_width + pad_width

                reconstructed = np.zeros((padded_height, padded_width, channels), dtype=np.uint8)
                index = 0
                for i in range(0, padded_height, block_height):
                    for j in range(0, padded_width, block_width):
                        block = codebook[labels[index]].reshape(block_height, block_width, channels)
                        reconstructed[i:i + block_height, j:j + block_width] = block
                        index += 1

                final_image = reconstructed[:original_height, :original_width]

                Image.fromarray(final_image).save(output_name)

                # Display preview
                preview_img = Image.fromarray(final_image)
                self.display_preview(preview_img, self.decompress_preview, 350)

                self.update_status(self.decompress_status, f">>> Decompression complete! Saved as '{output_name}'",
                                   self.success)

            except Exception as e:
                self.update_status(self.decompress_status, f"✗ Error: {str(e)}", self.error)
            finally:
                self.decompress_btn.config(state="normal")

        threading.Thread(target=decompress_thread, daemon=True).start()


if __name__ == "__main__":
    root = tk.Tk()
    app = VectorQuantizerGUI(root)
    root.mainloop()
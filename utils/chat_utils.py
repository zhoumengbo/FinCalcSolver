import tkinter as tk
from tkinter import scrolledtext
import GPUtil


def truncate_on_line_loop_threshold(s, is_json, loop_threshold, max_length):
    if is_json:
        return s
    if len(s) < max_length:
        return s
    lines = s.strip().split('\n')
    line_counts = {}
    processed_lines = []

    for line in lines:
        line_counts[line] = line_counts.get(line, 0) + 1

        if line_counts[line] > loop_threshold:
            break

        processed_lines.append(line)

    result = '\n'.join(processed_lines)

    if len(result) > max_length:
        return result[:max_length]
    else:
        return result


def get_gpu_memory():
    gpus = GPUtil.getGPUs()
    gpu_memory = [{'GPU': gpu.id, 'Memory Used': gpu.memoryUsed, 'Memory Total': gpu.memoryTotal} for gpu in gpus]
    return gpu_memory


def get_input():
    popup = tk.Tk()
    popup.title("Input Text")
    custom_font = ("Cambria", 12)
    text_widget = scrolledtext.ScrolledText(popup, wrap=tk.WORD, width=80, height=20, font=custom_font)
    text_widget.pack()
    user_input = tk.StringVar()

    def retrieve_text():
        input_text = text_widget.get("1.0", "end-1c")
        user_input.set(input_text)
        popup.destroy()

    button = tk.Button(popup, text="Get Text", command=retrieve_text)
    button.pack()
    popup.mainloop()

    return user_input.get()
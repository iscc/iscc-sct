"""
Gradio demo showcasing ISCC Semantic Text Code.

The demo features:

- two side by side text inputs.
- One sample text per input (One sample in english and the other a german translation of it)
- One slider to set global bitlength (32-256 bits in steps of 32 with 64 as default)
- One result output per text input

The user can select the samples or write or paste text into the inputs and generate ISCC Semantic
Text Codes for the Texts. Below the result outputs we show the similarity of the two codes.
"""

from loguru import logger as log
import gradio as gr
import iscc_sct as sct


def compute_iscc_code(text1, text2, bit_length):
    code1 = sct.gen_text_code_semantic(text1, bits=bit_length)
    code2 = sct.gen_text_code_semantic(text2, bits=bit_length)
    similarity = compare_codes(code1["iscc"], code2["iscc"], bit_length)
    return code1["iscc"], code2["iscc"], similarity


def compare_codes(code_a, code_b, bits):
    if all([code_a, code_b]):
        return generate_similarity_bar(hamming_to_cosine(sct.iscc_distance(code_a, code_b), bits))


def hamming_to_cosine(hamming_distance: int, dim: int) -> float:
    """Aproximate the cosine similarity for a given hamming distance and dimension"""
    result = 1 - (2 * hamming_distance) / dim
    return result


def generate_similarity_bar(similarity):
    """Generate a horizontal bar representing the similarity value, scaled to -100% to +100%."""
    # Scale similarity from [-1, 1] to [-100, 100]
    display_similarity = similarity * 100

    # Calculate the width of the bar based on the absolute value of similarity
    bar_width = int(abs(similarity) * 50)  # 50% is half the width of the container

    # Determine the color and starting position based on the sign of the similarity
    color = "green" if similarity >= 0 else "red"
    position = "left" if similarity >= 0 else "right"

    # Adjust the text position to be centered within the colored bar
    text_position = "left: 50%;" if similarity >= 0 else "right: 50%;"
    text_alignment = "transform: translateX(-50%);" if similarity >= 0 else "transform: translateX(50%);"

    bar_html = f"""
    <h3>Semantic Similarity</h3>
    <div style='width: 100%; border: 1px solid #ccc; height: 30px; position: relative; background-color: #eee;'>
        <div style='height: 100%; width: {bar_width}%; background-color: {color}; position: absolute; {position}: 50%;'>
            <span style='position: absolute; width: 100%; {text_position} top: 0; line-height: 30px; color: white; {text_alignment}'>{display_similarity:.2f}%</span>
        </div>
    </div>
    """
    return bar_html


# Sample texts
sample_text_en = "This is a sample text in English to demonstrate the ISCC-CODE generation."
sample_text_de = "Dies ist ein Beispieltext auf Deutsch, um die Erzeugung von ISCC-CODES zu demonstrieren."

custom_css = """
#chunked-text span.label {
    text-transform: none !important;
}
"""

iscc_theme = gr.themes.Default(
    font=[gr.themes.GoogleFont("Readex Pro")],
    font_mono=[gr.themes.GoogleFont("JetBrains Mono")],
    radius_size=gr.themes.sizes.radius_none,
)

with gr.Blocks(css=custom_css, theme=iscc_theme) as demo:
    with gr.Row(variant="panel"):
        gr.Markdown(
            """
        ## ✂️ ISCC Semantic Text-Code
        Demo of cross-lingual Semantic Text-Code (proof of concept)
        """,
        )
    with gr.Row(variant="panel"):
        in_iscc_bits = gr.Slider(
            label="ISCC Bit-Length",
            info="NUMBER OF BITS FOR OUTPUT ISCC",
            minimum=64,
            maximum=256,
            step=32,
            value=64,
        )
    with gr.Row(variant="panel"):
        with gr.Column(variant="panel"):
            in_text_a = gr.TextArea(
                label="Text",
                placeholder="Paste your text here or select sample from below",
                lines=12,
                max_lines=12,
            )

            gr.Examples(label="Sample Text", examples=[sample_text_en], inputs=[in_text_a])
            out_code_a = gr.Textbox(label="ISCC Code for Text A")
        with gr.Column(variant="panel"):
            in_text_b = gr.TextArea(
                label="Text",
                placeholder="Paste your text here or select sample from below",
                lines=12,
                max_lines=12,
            )

            gr.Examples(label="Sample Text", examples=[sample_text_de], inputs=[in_text_b])
            out_code_b = gr.Textbox(label="ISCC Code for Text B")

    with gr.Row(variant="panel"):
        with gr.Column(variant="panel"):
            out_similarity = gr.HTML(label="Similarity")

    def process_text(text, nbits, suffix):
        log.debug(f"{text[:20]}")
        if not text:
            return
        out_code_func = globals().get(f"out_code_{suffix}")
        iscc = sct.Metadata(**sct.gen_text_code_semantic(text, bits=nbits))
        result = {out_code_func: gr.Textbox(value=iscc.iscc)}
        return result

    in_text_a.change(
        lambda text, nbits: process_text(text, nbits, "a"),
        inputs=[in_text_a, in_iscc_bits],
        outputs=[out_code_a],
        show_progress="full",
    )
    in_text_b.change(
        lambda text, nbits: process_text(text, nbits, "b"),
        inputs=[in_text_b, in_iscc_bits],
        outputs=[out_code_b],
        show_progress="full",
    )

    out_code_a.change(compare_codes, inputs=[out_code_a, out_code_b, in_iscc_bits], outputs=[out_similarity])
    out_code_b.change(compare_codes, inputs=[out_code_a, out_code_b, in_iscc_bits], outputs=[out_similarity])
    with gr.Row():
        gr.ClearButton(components=[in_text_a, in_text_b])


if __name__ == "__main__":  # pragma: no cover
    demo.launch()

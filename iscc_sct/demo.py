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
import textwrap


def compute_iscc_code(text1, text2, bit_length):
    code1 = sct.gen_text_code_semantic(text1, bits=bit_length)
    code2 = sct.gen_text_code_semantic(text2, bits=bit_length)
    similarity = compare_codes(code1["iscc"], code2["iscc"], bit_length)
    return code1["iscc"], code2["iscc"], similarity


def compare_codes(code_a, code_b, bits):
    if all([code_a, code_b]):
        return generate_similarity_bar(hamming_to_cosine(sct.iscc_distance(code_a, code_b), bits))


def truncate_text(text, max_length=70):
    return textwrap.shorten(text, width=max_length, placeholder="...")


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
sample_text_en = "\n\n".join(
    [
        " ".join(paragraph.split())
        for paragraph in """
This document specifies the syntax and structure of the International Standard Content Code (ISCC),
as an identification system for digital assets (including encodings of text, images, audio, video or other content
across all media sectors). It also describes ISCC metadata and the use of ISCC in conjunction with other schemes, such
as DOI, ISAN, ISBN, ISRC, ISSN and ISWC.

An ISCC applies to a specific digital asset and is a data-descriptor deterministically constructed from multiple hash
digests using the algorithms and rules in this document. This document does not provide information on registration of
ISCCs.
""".strip().split("\n\n")
    ]
)

sample_text_de = "\n\n".join(
    [
        " ".join(paragraph.split())
        for paragraph in """
Dieses Dokument spezifiziert die Syntax und Struktur des International Standard Content Code (ISCC) als
Identifizierungssystem für digitale Inhalte (einschließlich Kodierungen von Text, Bildern, Audio, Video oder anderen
Inhalten in allen Medienbereichen). Sie beschreibt auch ISCC-Metadaten und die Verwendung von ISCC in Verbindung mit
anderen Systemen wie DOI, ISAN, ISBN, ISRC, ISSN und ISWC.

Ein ISCC bezieht sich auf ein bestimmtes digitales Gut und ist ein Daten-Deskriptor, der deterministisch aus mehreren
Hash-Digests unter Verwendung der Algorithmen und Regeln in diesem Dokument erstellt wird. Dieses Dokument enthält
keine Informationen über die Registrierung von ISCCs.
""".strip().split("\n\n")
    ]
)

custom_css = """
#chunked-text span.label {
    text-transform: none !important;
}

.clickable-example {
    cursor: pointer;
    transition: all 0.3s ease;
}

.clickable-example:hover {
    background-color: #f0f0f0;
    transform: scale(1.02);
}

.clickable-example .label-wrap {
    font-weight: bold;
    color: #4a90e2;
}

.truncate-text {
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    max-width: 300px;
    display: inline-block;
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
            value=128,
        )
    with gr.Row(variant="panel"):
        with gr.Column(variant="panel"):
            in_text_a = gr.TextArea(
                label="Text A",
                placeholder="Click the sample text below or type or paste your text.",
                lines=12,
                max_lines=12,
            )

            gr.Examples(
                label="Click to use sample text",
                examples=[[truncate_text(sample_text_en)]],
                inputs=[in_text_a],
                examples_per_page=1,
            )
            out_code_a = gr.Textbox(label="ISCC Code for Text A")
            gr.ClearButton(components=[in_text_a])
        with gr.Column(variant="panel"):
            in_text_b = gr.TextArea(
                label="Text B",
                placeholder="Click the sample text below or type or paste your text.",
                lines=12,
                max_lines=12,
            )

            gr.Examples(
                label="Click to use sample text",
                examples=[[truncate_text(sample_text_de)]],
                inputs=[in_text_b],
                examples_per_page=1,
            )
            out_code_b = gr.Textbox(label="ISCC Code for Text B")
            gr.ClearButton(components=[in_text_b])

    with gr.Row(variant="panel"):
        with gr.Column(variant="panel"):
            out_similarity = gr.HTML(label="Similarity")

    def process_text(text, nbits, suffix):
        log.debug(f"{text[:20]}")
        if not text:
            return None, text
        # Use the full sample text if it matches the truncated version
        full_text = (
            sample_text_en
            if text == truncate_text(sample_text_en)
            else sample_text_de
            if text == truncate_text(sample_text_de)
            else text
        )
        iscc = sct.Metadata(**sct.gen_text_code_semantic(full_text, bits=nbits))
        return iscc.iscc, full_text

    def recalculate_iscc(text_a, text_b, nbits):
        code_a = sct.gen_text_code_semantic(text_a, bits=nbits)["iscc"] if text_a else None
        code_b = sct.gen_text_code_semantic(text_b, bits=nbits)["iscc"] if text_b else None

        if code_a and code_b:
            similarity = compare_codes(code_a, code_b, nbits)
        else:
            similarity = None

        return (
            gr.Textbox(value=code_a) if code_a else gr.Textbox(),
            gr.Textbox(value=code_b) if code_b else gr.Textbox(),
            similarity,
        )

    in_text_a.change(
        process_text,
        inputs=[in_text_a, in_iscc_bits],
        outputs=[out_code_a, in_text_a],
        show_progress="full",
    )
    in_text_b.change(
        process_text,
        inputs=[in_text_b, in_iscc_bits],
        outputs=[out_code_b, in_text_b],
        show_progress="full",
    )

    in_iscc_bits.change(
        recalculate_iscc,
        inputs=[in_text_a, in_text_b, in_iscc_bits],
        outputs=[out_code_a, out_code_b, out_similarity],
        show_progress="full",
    )

    out_code_a.change(compare_codes, inputs=[out_code_a, out_code_b, in_iscc_bits], outputs=[out_similarity])
    out_code_b.change(compare_codes, inputs=[out_code_a, out_code_b, in_iscc_bits], outputs=[out_similarity])
    with gr.Row(variant="panel"):
        with gr.Column(variant="panel"):
            gr.Markdown(
"""
## Understanding ISCC Semantic Text-Codes

### What is an ISCC Semantic Text-Code?
An ISCC Semantic Text-Code is a digital fingerprint for text content. It captures the meaning of the text,
not just the exact words.

### How does it work?
1. **Input**: You provide a text in any language.
2. **Processing**: Our system analyzes the meaning of the text.
3. **Output**: A unique code is generated that represents the text's content.

### What can it do?
- **Cross-language matching**: It can recognize similar content across different languages.
- **Similarity detection**: It can measure how similar two texts are in meaning, not just in words.
- **Content identification**: It can help identify texts with similar content, even if the wording is different.

### How to use this demo:
1. **Enter text**: Type or paste text into either or both text boxes.
2. **Adjust bit length**: Use the slider to change the detail level of the code (higher = more detailed).
3. **View results**: See the generated ISCC code for each text.
4. **Compare**: Look at the similarity bar to see how alike the two texts are in meaning.

### Why is this useful?
- **Content creators**: Find similar content across languages.
- **Researchers**: Quickly compare documents or find related texts in different languages.
- **Publishers**: Identify potential translations or similar works efficiently.

This technology opens up new possibilities for understanding and managing text content across language barriers!
"""
            )

if __name__ == "__main__":  # pragma: no cover
    demo.launch()

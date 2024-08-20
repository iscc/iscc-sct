"""
Gradio demo showcasing ISCC Semantic Text Code.
"""

from loguru import logger as log
import gradio as gr
import iscc_sct as sct
import textwrap
import yaml
import pathlib


HERE = pathlib.Path(__file__).parent.absolute()


custom_css = """
.simbar {
    background: white;
    min-height: 30px;
}
"""


newline_symbols = {
    "\u000a": "⏎",  # Line Feed - Represented by the 'Return' symbol
    "\u000b": "↨",  # Vertical Tab - Represented by the 'Up Down Arrow' symbol
    "\u000c": "␌",  # Form Feed - Unicode Control Pictures representation
    "\u000d": "↵",  # Carriage Return - 'Downwards Arrow with Corner Leftwards' symbol
    "\u0085": "⤓",  # Next Line - 'Downwards Arrow with Double Stroke' symbol
    "\u2028": "↲",  # Line Separator - 'Downwards Arrow with Tip Leftwards' symbol
    "\u2029": "¶",  # Paragraph Separator - Represented by the 'Pilcrow' symbol
}


def no_nl(text):
    """Replace non-printable newline characters with printable symbols"""
    for char, symbol in newline_symbols.items():
        text = text.replace(char, symbol)
    return text


def no_nl_inner(text):
    """Replace non-printable newline characters with printable symbols, ignoring leading and
    trailing newlines"""
    # Strip leading and trailing whitespace
    stripped_text = text.strip()

    # Replace newline characters within the text
    for char, symbol in newline_symbols.items():
        stripped_text = stripped_text.replace(char, symbol)

    # Add back the leading and trailing newlines
    leading_newlines = len(text) - len(text.lstrip())
    trailing_newlines = len(text) - len(text.rstrip())

    return "\n" * leading_newlines + stripped_text + "\n" * trailing_newlines


def clean_chunk(chunk):
    """Strip consecutive line breaks in text to a maximum of 2."""
    return chunk.replace("\n\n", "\n")


def compute_iscc_code(text1, text2, bit_length):
    code1 = sct.gen_text_code_semantic(text1, bits=bit_length)
    code2 = sct.gen_text_code_semantic(text2, bits=bit_length)
    similarity = compare_codes(code1["iscc"], code2["iscc"], bit_length)
    return code1["iscc"], code2["iscc"], similarity


import binascii


def compare_codes(code_a, code_b, bits):
    if code_a and code_b:
        code_a_str = code_a.value if hasattr(code_a, "value") else str(code_a)
        code_b_str = code_b.value if hasattr(code_b, "value") else str(code_b)
        if code_a_str and code_b_str:
            try:
                distance = sct.iscc_distance(code_a_str, code_b_str)
                return generate_similarity_bar(hamming_to_cosine(distance, bits))
            except binascii.Error:
                # Invalid ISCC code format
                return None
    return None


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
    text_alignment = (
        "transform: translateX(-50%);" if similarity >= 0 else "transform: translateX(50%);"
    )

    tooltip = "Similarity based on ISCC code comparison, not direct text comparison."

    bar_html = f"""
    <div title="{tooltip}" style='width: 100%; border: 1px solid #ccc; height: 30px; position: relative; background-color: #eee;'>
        <div style='height: 100%; width: {bar_width}%; background-color: {color}; position: absolute; {position}: 50%;'>
            <span style='position: absolute; width: 100%; {text_position} top: 0; line-height: 30px; color: white; {text_alignment}'>{display_similarity:.2f}%</span>
        </div>
    </div>
    """
    return bar_html


def load_samples():
    with open(HERE / "samples.yml", "r", encoding="utf-8") as file:
        return yaml.safe_load(file)["samples"]


samples = load_samples()


iscc_theme = gr.themes.Default(
    font=[gr.themes.GoogleFont("Readex Pro Light")],
    font_mono=[gr.themes.GoogleFont("JetBrains Mono")],
    text_size=gr.themes.sizes.text_lg,
    radius_size=gr.themes.sizes.radius_none,
)

with gr.Blocks(css=custom_css, theme=iscc_theme) as demo:
    with gr.Row(variant="panel"):
        gr.Markdown(
            """
        ## 🔮️ ISCC - Semantic-Code Text
        Demo of cross-lingual Semantic Text-Code (proof of concept)
        """,
        )
    with gr.Row(variant="panel"):
        with gr.Column(variant="panel"):
            sample_dropdown_a = gr.Dropdown(
                choices=["None"] + [lang for lang in samples["a"]],
                label="Select sample for Text A",
                value="None",
            )
        with gr.Column(variant="panel"):
            sample_dropdown_b = gr.Dropdown(
                choices=["None"] + [lang for lang in samples["b"]],
                label="Select sample for Text B",
                value="None",
            )

    with gr.Row(variant="panel"):
        with gr.Column(variant="panel"):
            in_text_a = gr.TextArea(
                label="Text A",
                placeholder="Choose sample text from the dropdown above or type or paste your text.",
                lines=12,
                max_lines=12,
            )
            out_code_a = gr.Textbox(label="ISCC-SCT for Text A")
        with gr.Column(variant="panel"):
            in_text_b = gr.TextArea(
                label="Text B",
                placeholder="Choose sample text from the dropdown above or type or paste your text.",
                lines=12,
                max_lines=12,
            )
            out_code_b = gr.Textbox(label="ISCC-SCT for Text B")

    with gr.Row(variant="panel"):
        with gr.Column(variant="panel"):
            out_similarity_title = gr.Markdown("### ISCC-based Semantic Similarity")
            with gr.Row(elem_classes="simbar"):
                out_similarity = gr.HTML()
            gr.Markdown(
                "**NOTE:** Similarity is calculated based on the generated ISCC-SCT, not the original text."
            )

    with gr.Row(variant="panel"):
        reset_button = gr.Button("Reset All")

    with gr.Accordion(label="🔍 Explore Details & Advanced Options", open=True):
        with gr.Row(variant="panel"):
            with gr.Column(variant="panel"):
                in_iscc_bits = gr.Slider(
                    label="ISCC Bit-Length",
                    info="NUMBER OF BITS FOR OUTPUT ISCC",
                    minimum=64,
                    maximum=256,
                    step=32,
                    value=sct.sct_opts.bits,
                )
            with gr.Column(variant="panel"):
                in_max_tokens = gr.Slider(
                    label="Max Tokens",
                    info="MAXIMUM NUMBER OF TOKENS PER CHUNK",
                    minimum=49,
                    maximum=sct.sct_opts.max_tokens,
                    step=1,
                    value=127,
                )

        with gr.Row(variant="panel"):
            with gr.Column(variant="panel"):
                out_chunks_a = gr.HighlightedText(
                    label="Chunked Text A",
                    interactive=False,
                    elem_id="chunked-text-a",
                )
            with gr.Column(variant="panel"):
                out_chunks_b = gr.HighlightedText(
                    label="Chunked Text B",
                    interactive=False,
                    elem_id="chunked-text-b",
                )

        with gr.Row(variant="panel"):
            with gr.Column(variant="panel"):
                gr.Markdown("### Granular Matches")
                in_granular_matches = gr.Dataframe(
                    headers=["Chunk A", "Similarity", "Chunk B"],
                    column_widths=["45%", "10%", "45%"],
                    wrap=True,
                    elem_classes="granular-matches",
                )

    def update_sample_text(choice, group):
        if choice == "None":
            return ""
        return samples[group][choice]

    sample_dropdown_a.change(
        lambda choice: update_sample_text(choice, "a"),
        inputs=[sample_dropdown_a],
        outputs=[in_text_a],
    )
    sample_dropdown_b.change(
        lambda choice: update_sample_text(choice, "b"),
        inputs=[sample_dropdown_b],
        outputs=[in_text_b],
    )

    def process_and_calculate(text_a, text_b, nbits, max_tokens):
        log.debug(f"Processing text_a: {text_a[:20]}, text_b: {text_b[:20]}")

        def process_single_text(text, suffix):
            out_code_func = globals().get(f"out_code_{suffix}")
            out_chunks_func = globals().get(f"out_chunks_{suffix}")

            if not text:
                return {
                    out_code_func: gr.Textbox(value=None),
                    out_chunks_func: gr.HighlightedText(
                        value=None, elem_id=f"chunked-text-{suffix}"
                    ),
                }

            result = sct.gen_text_code_semantic(
                text,
                bits=nbits,
                simprints=True,
                offsets=True,
                sizes=True,
                contents=True,
                max_tokens=max_tokens,
            )
            iscc = sct.Metadata(**result).to_object_format()

            # Generate chunked text with simprints and overlaps
            features = iscc.features[0]
            highlighted_chunks = []
            overlaps = iscc.get_overlaps()

            for i, feature in enumerate(features.simprints):
                feature: sct.Feature
                content = feature.content

                # Remove leading overlap
                if i > 0 and overlaps[i - 1]:
                    content = content[len(overlaps[i - 1]) :]

                # Remove trailing overlap
                if i < len(overlaps) and overlaps[i]:
                    content = content[: -len(overlaps[i])]

                label = f"{feature.size}:{feature.simprint}"
                highlighted_chunks.append((no_nl_inner(content), label))

                if i < len(overlaps):
                    overlap = overlaps[i]
                    if overlap:
                        highlighted_chunks.append((f"\n{no_nl(overlap)}\n", "overlap"))

            return {
                out_code_func: gr.Textbox(value=iscc.iscc),
                out_chunks_func: gr.HighlightedText(
                    value=highlighted_chunks, elem_id=f"chunked-text-{suffix}"
                ),
                "metadata": iscc,
            }

        result_a = process_single_text(text_a, "a")
        result_b = process_single_text(text_b, "b")

        code_a = result_a[out_code_a] if text_a else None
        code_b = result_b[out_code_b] if text_b else None

        similarity = compare_codes(code_a, code_b, nbits) or out_similarity

        granular_matches = []
        if text_a and text_b:
            matches = sct.granular_similarity(
                result_a["metadata"], result_b["metadata"], threshold=80
            )
            for match in matches:
                granular_matches.append(
                    [
                        match[0].content,
                        f"{match[1]}%",
                        match[2].content,
                    ]
                )

        return (
            result_a[out_code_a],
            result_a[out_chunks_a],
            result_b[out_code_b],
            result_b[out_chunks_b],
            similarity,
            gr.Dataframe(value=granular_matches),
        )

    in_text_a.change(
        process_and_calculate,
        inputs=[in_text_a, in_text_b, in_iscc_bits, in_max_tokens],
        outputs=[
            out_code_a,
            out_chunks_a,
            out_code_b,
            out_chunks_b,
            out_similarity,
            in_granular_matches,
        ],
        show_progress="full",
        trigger_mode="always_last",
    )

    in_text_b.change(
        process_and_calculate,
        inputs=[in_text_a, in_text_b, in_iscc_bits, in_max_tokens],
        outputs=[
            out_code_a,
            out_chunks_a,
            out_code_b,
            out_chunks_b,
            out_similarity,
            in_granular_matches,
        ],
        show_progress="full",
        trigger_mode="always_last",
    )

    in_iscc_bits.change(
        process_and_calculate,
        inputs=[in_text_a, in_text_b, in_iscc_bits, in_max_tokens],
        outputs=[
            out_code_a,
            out_chunks_a,
            out_code_b,
            out_chunks_b,
            out_similarity,
            in_granular_matches,
        ],
        show_progress="full",
    )

    in_max_tokens.change(
        process_and_calculate,
        inputs=[in_text_a, in_text_b, in_iscc_bits, in_max_tokens],
        outputs=[
            out_code_a,
            out_chunks_a,
            out_code_b,
            out_chunks_b,
            out_similarity,
            in_granular_matches,
        ],
        show_progress="full",
    )

    out_code_a.change(
        compare_codes, inputs=[out_code_a, out_code_b, in_iscc_bits], outputs=[out_similarity]
    )
    out_code_b.change(
        compare_codes, inputs=[out_code_a, out_code_b, in_iscc_bits], outputs=[out_similarity]
    )

    def reset_all():
        return (
            gr.Slider(value=64),  # Reset ISCC Bit-Length
            gr.Dropdown(
                value="None", choices=["None"] + [lang for lang in samples["a"]]
            ),  # Reset sample dropdown A
            gr.Dropdown(
                value="None", choices=["None"] + [lang for lang in samples["b"]]
            ),  # Reset sample dropdown B
            gr.TextArea(value=""),  # Reset Text A
            gr.TextArea(value=""),  # Reset Text B
            gr.Textbox(value=""),  # Reset ISCC Code for Text A
            gr.Textbox(value=""),  # Reset ISCC Code for Text B
            gr.HTML(value=""),  # Reset Similarity
            gr.HighlightedText(value=[]),  # Reset Chunked Text A
            gr.HighlightedText(value=[]),  # Reset Chunked Text B
        )

    reset_button.click(
        reset_all,
        outputs=[
            in_iscc_bits,
            sample_dropdown_a,
            sample_dropdown_b,
            in_text_a,
            in_text_b,
            out_code_a,
            out_code_b,
            out_similarity,
            out_chunks_a,
            out_chunks_b,
        ],
    )

    with gr.Row(variant="panel"):
        gr.Markdown(
            """
## Understanding ISCC Semantic Text-Codes

### What is an ISCC Semantic Text-Code?
An ISCC Semantic Text-Code is a digital fingerprint for text content. It captures the meaning of
the text, not just the exact words. Technically it is am ISCC-encoded, binarized multi-lingual
document-embedding.

### How does it work?
1. **Input**: You provide a text in any language.
2. **Processing**: Vector embeddings are created for individual chunks of the text.
3. **Output**: A unique ISCC-UNIT is generated that represents the entire text's content.

### What can it do?
- **Cross-language matching**: It can recognize similar content across different languages.
- **Similarity detection**: It can measure how similar two texts are in meaning, not just in words.
- **Content identification**: It can help identify texts with similar content, even if the wording
    is different.

### How to use this demo:
1. **Enter text**: Type or paste text into either or both text boxes.
2. **Adjust bit length**: Use the slider to change the detail level of the code (higher = more
    detailed).
3. **View results**: See the generated ISCC code for each text.
4. **Compare**: Look at the similarity bar to see how alike the two texts are in meaning, based on
    their ISCC codes.

### Important Note:
The similarity shown is calculated by comparing the ISCC codes, not the original texts. This
allows for efficient and privacy-preserving comparisons, as only the codes need to be shared
or stored.
"""
        )

        gr.Markdown(
            """
### Why is this useful?
- **Content creators**: Find similar content across languages.
- **Researchers**: Quickly compare documents or find related texts in different languages.
- **Publishers**: Identify potential translations or similar works efficiently.

This technology opens up new possibilities for understanding and managing text content across
language barriers!

### Explore Details & Advanced Options

The "Explore Details & Advanced Options" section provides additional tools and information:

1. **ISCC Bit-Length**: Adjust the precision of the ISCC code. Higher values provide more detailed
comparisons but may be more sensitive to minor differences.

2. **Max Tokens**: Set the maximum number of tokens per chunk. This affects how the text is split
for processing.

3. **Chunked Text**: View how each input text is divided into chunks for processing. Each chunk is
color-coded and labeled with its size and simprint (a similarity preserving fingerprint).

4. **Granular Matches**: See a detailed comparison of individual chunks between Text A and Text B.
This table shows which specific parts of the texts are most similar (above 80%), along with their
approximate cosine similarity (scaled -100% to +100%).

For more information about the **ISCC** see:
- https://github.com/iscc
- https://iscc.codes
- https://iscc.io
- [ISO 24138:2024](https://www.iso.org/standard/77899.html)
"""
        )
    with gr.Row():
        gr.Markdown(
            f"iscc-sct v{sct.__version__} | Source Code: https://github.com/iscc/iscc-sct",
            elem_classes="footer",
        )

if __name__ == "__main__":  # pragma: no cover
    demo.launch()

from html import escape
import iscc_sct as ict


def generate_html(fingerprint_data):
    chunks = fingerprint_data["features"]

    # Sort chunks by offset
    chunks.sort(key=lambda x: x["offset"])

    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Text Fingerprint Visualization</title>
        <script src="https://cdn.tailwindcss.com"></script>
    </head>
    <body class="bg-gray-100 p-8">
        <div class="max-w-4xl mx-auto bg-white p-6 rounded-lg shadow-lg">
            <h1 class="text-2xl font-bold mb-4">Text Fingerprint Visualization</h1>
            <div class="text-sm mb-4">
                <span class="font-semibold">ISCC:</span> {fingerprint_data['iscc']}
            </div>
            <div class="text-sm mb-4">
                <span class="font-semibold">Characters:</span> {fingerprint_data['characters']}
            </div>
            <div class="relative text-base leading-relaxed whitespace-pre-wrap">
    """

    chunk_color = "bg-yellow-100"
    overlap_color = "bg-red-100"

    current_pos = 0
    for i, chunk in enumerate(chunks):
        start = max(chunk["offset"], current_pos)
        end = chunk["offset"] + chunk["size"]

        if start < end:
            # Function to escape text and preserve line breaks
            def escape_and_preserve_breaks(text):
                return escape(text).replace("\n", "<br>")

            # Non-overlapping part
            html_content += f'<span class="{overlap_color}">{escape_and_preserve_breaks(chunk["text"][current_pos - chunk["offset"]:start - chunk["offset"]])}'

            # Overlapping part (if any)
            if i < len(chunks) - 1 and end > chunks[i + 1]["offset"]:
                overlap_end = chunks[i + 1]["offset"]
                html_content += f'<span class="{chunk_color}">{escape_and_preserve_breaks(chunk["text"][start - chunk["offset"]:overlap_end - chunk["offset"]])}</span>'
                html_content += escape_and_preserve_breaks(
                    chunk["text"][overlap_end - chunk["offset"] :]
                )
            else:
                html_content += escape_and_preserve_breaks(chunk["text"][start - chunk["offset"] :])

            # Fingerprint badge
            html_content += f'<span class="inline-block bg-gray-800 text-white text-xs px-2 py-1 rounded ml-1">{chunk["feature"]}</span>'

            html_content += "</span>"

            current_pos = end

    html_content += """
            </div>
        </div>
    </body>
    </html>
    """
    return html_content


def main():
    with open("../README.md", "rb") as f:
        data = f.read()

    text = data.decode("utf-8")

    result = ict.create(text, granular=True)
    print(result.model_dump())

    # Generate the HTML content
    html_content = generate_html(result.model_dump())

    # Write the HTML content to a file
    with open("readme.html", "wt", encoding="utf-8") as f:
        f.write(html_content)


if __name__ == "__main__":
    main()

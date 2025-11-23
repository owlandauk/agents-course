import os
import sys
import re
from huggingface_hub import InferenceClient

# Get the directory containing the current script
script_dir = os.path.dirname(os.path.abspath(__file__))
# Normalize default input directory (resolve '..' and use native separators)
default_inp_dir = os.path.normpath(os.path.join(script_dir, '..', 'units', 'en'))
default_model = "deepseek-ai/DeepSeek-R1"
default_client = InferenceClient(
	provider="together",
    # api_key is read from the environment
)

def auto_translate(
    output_lang: str,
    prompt: callable,
    inp_dir: str = default_inp_dir,
    model: str = default_model,
    client: InferenceClient = default_client
):
    # Build the output path by computing the path relative to the input directory
    def get_output_path(inp_path: str) -> str:
        rel = os.path.relpath(inp_path, inp_dir)
        base_dir = os.path.dirname(inp_dir)
        out_path = os.path.normpath(os.path.join(base_dir, output_lang, rel))
        return out_path
    escape_special_tokens = lambda x: x.replace('<think>', '<%%think%%>').replace('</think>', '<%%/think%%>')
    unescape_special_tokens = lambda x: x.replace('<%%think%%>', '<think>').replace('<%%/think%%>', '</think>')

    # Get the list of all files in the directory, recursively
    inp_files: list[str] = []
    print('Collecting files...')
    for root, dirs, files in os.walk(inp_dir):
        for file in files:
            if file.endswith('.mdx') or file == "_toctree.yml":
                fname = os.path.join(root, file)
                print('  +', fname)
                inp_files.append(fname)

    def write_out_file(fpath: str, content: str):
        base_path = os.path.dirname(fpath)
        os.makedirs(base_path, exist_ok=True)
        with open(fpath, 'w', encoding='utf-8') as f:
            f.write(content)

    # Read the content of the file and process
    for i, inp_file in enumerate(inp_files):
        out_file = get_output_path(inp_file)
        if os.path.exists(out_file):
            print(f'[{i+1}/{len(inp_files)}] Skipping file: {inp_file}')
            continue
        with open(inp_file, 'r', encoding='utf-8') as f:
            content: str = f.read()
            content = escape_special_tokens(content)
            if content.strip() == "":
                print(f'[{i+1}/{len(inp_files)}] Skipping empty file: {inp_file}')
                write_out_file(out_file, "")
                continue

            print(f'[{i+1}/{len(inp_files)}] Processing file: {inp_file}')
            try:
                stream = client.chat.completions.create(
                model=model,
                temperature=0.0,
                messages=[
                    {"role": "user", "content": prompt(content)},
                ],
                stream=True,
            )
            except Exception as e:
                print(f"Error creating stream for {inp_file}: {e}")
                continue

            final_text = ""
            # Stream might yield objects or dicts depending on client version.
            for chunk in stream:
                try:
                    # Try attribute access first
                    content_piece = getattr(chunk.choices[0].delta, 'content', None)
                except (AttributeError, IndexError, TypeError) as e:
                    print(f"Error accessing attribute content in chunk: {e}")
                    content_piece = None
                if content_piece is None:
                    try:
                        # Fallback to dict-style access
                    except (KeyError, TypeError, IndexError):
                    except (KeyError, TypeError, IndexError) as e:
                        print(f"Error accessing dict content in chunk: {e}")
                        content_piece = None

                if not content_piece:
                    continue

                print(content_piece, end="")
                sys.stdout.flush()
                final_text += content_piece
            final_text = final_text.split("</think>")[-1].strip()
            # Remove any model-inserted <think>...</think> reasoning blocks if present
            final_text = re.sub(r"<think>.*?</think>", "", final_text, flags=re.DOTALL).strip()
            # Write the output to the file
            final_text = unescape_special_tokens(final_text)
            write_out_file(out_file, final_text)
            print()
            print(f'  -> Translated to: {out_file}')
            print("--" * 20)
            #break

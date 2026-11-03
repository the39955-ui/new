"""
3D Animation Prompt & Workflow Generator
Gemini (multimodal) -> structured Blender scene prompts -> paste into Claude (Blender MCP)

Run:
    pip install streamlit google-genai
    streamlit run app.py
"""

import json
import os
import re
import time
import uuid

import streamlit as st
from google import genai
from google.genai import types

st.set_page_config(page_title="Blender Prompt Generator", page_icon="🎬", layout="wide")

DEFAULT_MODELS = [
    "gemini-3.1-pro-preview",
    "gemini-3.1-flash-lite",
    "gemini-flash-latest",
    "gemini-flash-lite-latest",
    "gemini-2.5-pro",
    "gemini-2.5-flash",
]

# --------------------------------------------------------------------------- #
# Session state helpers
# --------------------------------------------------------------------------- #
def new_segment(**kw):
    base = {
        "id": uuid.uuid4().hex[:8],
        "start": 0.0,
        "end": 6.0,
        "title": "",
        "description": "",
        "motion": "",
        "camera": "",
    }
    base.update(kw)
    return base


def new_character(**kw):
    base = {
        "id": uuid.uuid4().hex[:8],
        "name": "",
        "description": "",
        "asset_source": "",
    }
    base.update(kw)
    return base


def init_state():
    ss = st.session_state
    ss.setdefault("api_key", os.environ.get("GEMINI_API_KEY", ""))
    ss.setdefault("models", DEFAULT_MODELS)
    ss.setdefault(
        "characters",
        [
            new_character(
                name="Moo-Moo",
                description="Friendly cartoon cow, black and white patches, pink snout, "
                "straw hat, big brown eyes, Pixar-style stylised 3D.",
            )
        ],
    )
    ss.setdefault(
        "segments",
        [
            new_segment(
                start=0.0,
                end=6.0,
                title="HOOK",
                description="Extreme close-up of Moo-Moo's nose on a sunny farm.",
                motion="Tongue licks the lens.",
                camera="Static extreme close-up, camera wobbles on the lick.",
            )
        ],
    )
    ss.setdefault("results", {})
    ss.setdefault("final_prompt", "")


init_state()
ss = st.session_state

# --------------------------------------------------------------------------- #
# Sidebar: API key + model
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.header("🔑 Gemini setup")

    key_input = st.text_input(
        "API key",
        value="",
        type="password",
        placeholder="Paste key, then press Save"
        if not ss.api_key
        else "Key saved (hidden). Paste to replace.",
    )
    c1, c2 = st.columns(2)
    if c1.button("Save key", use_container_width=True):
        if key_input.strip():
            ss.api_key = key_input.strip()
            st.success("Saved for this session.")
        else:
            st.warning("Enter a key first.")
    if c2.button("Clear key", use_container_width=True):
        ss.api_key = ""
        st.info("Key removed.")
    st.caption(
        "✅ Key stored in session memory only."
        if ss.api_key
        else "No key yet. You can also set the GEMINI_API_KEY env variable."
    )

    st.divider()
    st.header("🧠 Model")

    def refresh_models():
        try:
            client = genai.Client(api_key=ss.api_key)
            found = []
            for m in client.models.list():
                actions = getattr(m, "supported_actions", None) or []
                if "generateContent" in actions and "gemini" in m.name:
                    found.append(m.name.replace("models/", ""))
            if found:
                ss.models = sorted(set(found), reverse=True)
                st.success(f"Loaded {len(found)} models.")
            else:
                st.warning("No models returned.")
        except Exception as e:
            st.error(f"Could not fetch models: {e}")

    if st.button("↻ Refresh model list from API", use_container_width=True, disabled=not ss.api_key):
        refresh_models()

    model = st.selectbox("Gemini multimodal model", ss.models, index=0)
    temperature = st.slider("Temperature", 0.0, 1.5, 0.4, 0.1)

    st.divider()
    st.header("🎞️ Scene settings")
    fps = st.number_input("Frame rate (fps)", 12, 60, 24)
    style = st.text_input("Visual style", "Stylised Pixar-like 3D cartoon, bright, soft lighting")
    render_engine = st.selectbox("Render engine", ["EEVEE", "Cycles"])

# --------------------------------------------------------------------------- #
# Main layout
# --------------------------------------------------------------------------- #
st.title("🎬 3D Animation Prompt Generator")
st.caption("Describe scenes → Gemini structures them → copy into Claude to build the Blender script.")

# ---- Characters ---------------------------------------------------------- #
st.subheader("1 · Characters")
char_files = {}
for i, ch in enumerate(ss.characters):
    with st.expander(f"Character: {ch['name'] or 'Unnamed'}", expanded=(i == 0)):
        a, b = st.columns([2, 3])
        ch["name"] = a.text_input("Name", ch["name"], key=f"cn_{ch['id']}")
        ch["asset_source"] = a.text_input(
            "Existing asset (optional)",
            ch["asset_source"],
            key=f"ca_{ch['id']}",
            help="File path (.blend/.glb/.fbx), Sketchfab/Poly Haven link, or 'generate from image'.",
        )
        ch["description"] = b.text_area(
            "Look & personality", ch["description"], key=f"cd_{ch['id']}", height=110
        )
        char_files[ch["id"]] = st.file_uploader(
            "Reference images (turnaround, concept art)",
            type=["png", "jpg", "jpeg", "webp"],
            accept_multiple_files=True,
            key=f"cf_{ch['id']}",
        )
        if len(ss.characters) > 1 and st.button("Remove character", key=f"cr_{ch['id']}"):
            ss.characters.remove(ch)
            st.rerun()
if st.button("＋ Add character"):
    ss.characters.append(new_character())
    st.rerun()

st.text_area(
    "Global environment notes (optional)",
    key="env_notes",
    placeholder="e.g. Sunny farm, wooden fence, hay bales, blue sky with soft clouds.",
    height=80,
)

# ---- Segments ------------------------------------------------------------ #
st.subheader("2 · Scene segments")
seg_files = {}
for i, sg in enumerate(ss.segments):
    label = f"Segment {i + 1} · {sg['title'] or 'Untitled'} ({sg['start']:.1f}s–{sg['end']:.1f}s)"
    with st.expander(label, expanded=(i == 0)):
        t1, t2, t3 = st.columns([1, 1, 3])
        sg["start"] = t1.number_input("Start (s)", 0.0, 3600.0, float(sg["start"]), 0.5, key=f"ss_{sg['id']}")
        sg["end"] = t2.number_input("End (s)", 0.0, 3600.0, float(sg["end"]), 0.5, key=f"se_{sg['id']}")
        sg["title"] = t3.text_input("Title", sg["title"], key=f"st_{sg['id']}", placeholder="HOOK, REVEAL, CTA…")

        d1, d2 = st.columns(2)
        sg["description"] = d1.text_area(
            "Scene description", sg["description"], key=f"sd_{sg['id']}", height=110
        )
        sg["motion"] = d2.text_area(
            "Motion / action details", sg["motion"], key=f"sm_{sg['id']}", height=110
        )
        sg["camera"] = st.text_input("Camera notes", sg["camera"], key=f"sc_{sg['id']}")
        seg_files[sg["id"]] = st.file_uploader(
            "Storyboard panel / reference images (optional)",
            type=["png", "jpg", "jpeg", "webp"],
            accept_multiple_files=True,
            key=f"sf_{sg['id']}",
        )
        if len(ss.segments) > 1 and st.button("Remove segment", key=f"sr_{sg['id']}"):
            ss.segments.remove(sg)
            st.rerun()

if st.button("＋ Add segment"):
    last_end = ss.segments[-1]["end"] if ss.segments else 0.0
    ss.segments.append(new_segment(start=last_end, end=last_end + 6.0))
    st.rerun()

# --------------------------------------------------------------------------- #
# Gemini generation
# --------------------------------------------------------------------------- #
SYSTEM_PROMPT = """You are a technical director who converts animation storyboards into precise,
Blender-ready scene specifications. Another AI (Claude, using a Blender MCP connector) will read your
output and write a Blender Python (bpy) script, so be concrete and unambiguous.

Rules:
- Blender conventions: right-handed, Z-up, metres, rotations in degrees (Euler XYZ), camera looks down its local -Z.
- Every character must include import instructions (how to obtain/import the asset) AND exact placement
  (location, rotation, scale) plus a starting pose.
- Use the reference images (if any) to infer proportions, colours, materials, props and framing.
- Times are in seconds from the start of the whole animation; also give frame numbers at the stated fps.
- Keep animation as keyframe lists with explicit values; avoid vague words like "somewhat".
- Maintain continuity with the previous segment's end state when provided.
- Do not invent characters not listed, unless they are clearly visible in a storyboard image.
- Return ONLY valid JSON, no markdown fences."""

SCHEMA_HINT = """{
  "segment_title": str,
  "time_range": {"start_s": number, "end_s": number, "start_frame": int, "end_frame": int},
  "summary": str,
  "environment": {"setting": str, "objects": [{"name": str, "type": str, "location": [x,y,z], "rotation_deg": [x,y,z], "scale": [x,y,z]}], "lighting": str, "world_background": str},
  "characters": [{
     "name": str,
     "import_strategy": str,
     "placement": {"location": [x,y,z], "rotation_deg": [x,y,z], "scale": [x,y,z]},
     "initial_pose": str,
     "materials_notes": str,
     "rig_notes": str
  }],
  "camera": {"shot_type": str, "location": [x,y,z], "rotation_deg": [x,y,z], "focal_length_mm": number, "dof": {"enabled": bool, "focus_target": str, "f_stop": number}, "movement": str},
  "animation": [{"target": str, "action": str, "keyframes": [{"time_s": number, "frame": int, "property": str, "value": str}], "easing": str}],
  "claude_instructions": str,
  "end_state": str
}"""


def load_parts(files):
    parts = []
    for f in files or []:
        parts.append(types.Part.from_bytes(data=f.getvalue(), mime_type=f.type or "image/png"))
    return parts


def strip_fences(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text


def call_gemini(client, contents):
    cfg = types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
        temperature=temperature,
        response_mime_type="application/json",
    )
    last_err = None
    for attempt in range(3):
        try:
            resp = client.models.generate_content(model=model, contents=contents, config=cfg)
            return resp.text
        except Exception as e:  # noqa: BLE001
            last_err = e
            time.sleep(2 * (attempt + 1))
    raise last_err


def generate_all():
    client = genai.Client(api_key=ss.api_key)
    ss.results = {}
    prev_end_state = ""
    progress = st.progress(0.0, text="Starting…")
    total = len(ss.segments)

    char_block = "\n".join(
        f"- {c['name'] or 'Unnamed'}: {c['description']} | asset: {c['asset_source'] or 'none provided'}"
        for c in ss.characters
    )

    for idx, sg in enumerate(ss.segments):
        progress.progress(idx / total, text=f"Segment {idx + 1}/{total}: {sg['title'] or 'Untitled'}")
        contents = []

        # character reference images first
        for c in ss.characters:
            imgs = load_parts(char_files.get(c["id"]))
            if imgs:
                contents.append(f"Reference images for character '{c['name']}':")
                contents.extend(imgs)
        seg_imgs = load_parts(seg_files.get(sg["id"]))
        if seg_imgs:
            contents.append("Storyboard / reference images for THIS segment:")
            contents.extend(seg_imgs)

        contents.append(
            f"""Global settings:
- FPS: {fps}
- Visual style: {style}
- Render engine: {render_engine}
- Environment notes: {st.session_state.get('env_notes', '') or 'none'}

Characters:
{char_block}

Segment {idx + 1} of {total}
- Title: {sg['title']}
- Time: {sg['start']}s to {sg['end']}s
- Scene description: {sg['description']}
- Motion details: {sg['motion']}
- Camera notes: {sg['camera']}

Previous segment end state: {prev_end_state or 'n/a (first segment)'}

Return JSON matching this shape exactly:
{SCHEMA_HINT}"""
        )

        raw = call_gemini(client, contents)
        try:
            data = json.loads(strip_fences(raw))
            prev_end_state = data.get("end_state", "") if isinstance(data, dict) else ""
        except json.JSONDecodeError:
            data = {"_raw": raw, "_error": "Model did not return valid JSON"}
        ss.results[sg["id"]] = data

    progress.progress(1.0, text="Done")
    build_final_prompt()


def build_final_prompt():
    lines = []
    lines.append("# TASK: Build a Blender animation via the Blender MCP connector\n")
    lines.append(
        "Using the Blender connector, write and execute a Python (bpy) script that builds the scene "
        "below. Work segment by segment so I can review each one.\n"
    )
    lines.append("## Requirements")
    lines.append(f"- FPS: {fps}  |  Render engine: {render_engine}")
    lines.append("- Units: metres, Z-up. Rotations are in degrees (Euler XYZ).")
    lines.append(f"- Style: {style}")
    lines.append(
        "- Import each character using its `import_strategy` (local file, Sketchfab, Poly Haven, "
        "Hyper3D/Hunyuan3D, or a primitive stand-in if unavailable) and apply the exact placement.")
    lines.append("- Use keyframes (`keyframe_insert`) with the times/frames given; set scene frame range to cover all segments.")
    lines.append("- Name every object clearly so later segments can reference them.")
    lines.append("- After each segment, report what was created and any assumptions or fallbacks.\n")

    lines.append("## Characters")
    for c in ss.characters:
        lines.append(f"- **{c['name'] or 'Unnamed'}**: {c['description']}")
        if c["asset_source"]:
            lines.append(f"  - Asset: {c['asset_source']}")
    if st.session_state.get("env_notes"):
        lines.append(f"\n## Global environment\n{st.session_state['env_notes']}")

    lines.append("\n## Segments")
    for i, sg in enumerate(ss.segments, 1):
        data = ss.results.get(sg["id"])
        if not data:
            continue
        lines.append(f"\n### Segment {i}: {sg['title'] or 'Untitled'} ({sg['start']}s–{sg['end']}s)")
        if isinstance(data, dict) and "_raw" in data:
            lines.append(data["_raw"])
        else:
            if data.get("summary"):
                lines.append(data["summary"])
            if data.get("claude_instructions"):
                lines.append(f"\n**Instructions:** {data['claude_instructions']}")
            lines.append("\n```json")
            lines.append(json.dumps(data, indent=2, ensure_ascii=False))
            lines.append("```")

    lines.append("\n## Deliverable")
    lines.append(
        "Start with Segment 1: generate the bpy script, execute it through the Blender connector, "
        "and confirm the result before continuing."
    )
    ss.final_prompt = "\n".join(lines)


st.subheader("3 · Generate")
ready = bool(ss.api_key) and any(s["description"].strip() or s["motion"].strip() for s in ss.segments)
if not ss.api_key:
    st.info("Add and save your Gemini API key in the sidebar.")

if st.button("✨ Generate Blender prompts", type="primary", disabled=not ready):
    try:
        generate_all()
        st.success("Prompts generated.")
    except Exception as e:  # noqa: BLE001
        st.error(f"Generation failed: {e}")

# --------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------- #
if ss.results:
    st.subheader("4 · Review")
    for i, sg in enumerate(ss.segments, 1):
        data = ss.results.get(sg["id"])
        if data is None:
            continue
        with st.expander(f"Segment {i}: {sg['title'] or 'Untitled'}", expanded=False):
            if isinstance(data, dict) and "_raw" in data:
                st.warning(data["_error"])
                st.code(data["_raw"])
            else:
                st.json(data, expanded=1)

if ss.final_prompt:
    st.subheader("5 · Copy into Claude")
    st.caption("Use the copy icon at the top-right of the block, or download as a file.")
    st.code(ss.final_prompt, language="markdown", wrap_lines=True)
    st.download_button(
        "⬇️ Download as .md",
        ss.final_prompt,
        file_name="blender_prompt_for_claude.md",
        mime="text/markdown",
    )
    with st.expander("Edit before copying"):
        edited = st.text_area("Final prompt", ss.final_prompt, height=400)
        if st.button("Apply edits"):
            ss.final_prompt = edited
            st.rerun()

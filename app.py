"""
Storyboard -> Blender Prompt Generator
Upload a storyboard image. Gemini reads it, extracts characters / segments / timing,
then generates Blender-ready specs you paste into Claude (Blender MCP).

Run:
    pip install -r requirements.txt
    streamlit run app.py
"""

import json
import os
import re
import time

import streamlit as st
from google import genai
from google.genai import types

st.set_page_config(page_title="Storyboard → Blender Prompts", page_icon="🎬", layout="wide")

DEFAULT_MODELS = [
    "gemini-3.1-pro-preview",
    "gemini-3.1-flash-lite",
    "gemini-flash-latest",
    "gemini-flash-lite-latest",
    "gemini-2.5-pro",
    "gemini-2.5-flash",
]

ss = st.session_state
ss.setdefault("api_key", os.environ.get("GEMINI_API_KEY", ""))
ss.setdefault("models", DEFAULT_MODELS)
ss.setdefault("analysis", None)
ss.setdefault("results", [])
ss.setdefault("final_prompt", "")

# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.header("🔑 Gemini API key")
    key_input = st.text_input(
        "API key",
        type="password",
        placeholder="Key saved (hidden). Paste to replace." if ss.api_key else "Paste key, then Save",
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
    st.caption("✅ Key kept in session memory only." if ss.api_key else "No key saved yet.")

    st.divider()
    st.header("🧠 Model")
    if st.button("↻ Refresh model list from API", use_container_width=True, disabled=not ss.api_key):
        try:
            client = genai.Client(api_key=ss.api_key)
            found = [
                m.name.replace("models/", "")
                for m in client.models.list()
                if "generateContent" in (getattr(m, "supported_actions", None) or [])
                and "gemini" in m.name
            ]
            if found:
                ss.models = sorted(set(found), reverse=True)
                st.success(f"Loaded {len(found)} models.")
            else:
                st.warning("No models returned.")
        except Exception as e:  # noqa: BLE001
            st.error(f"Could not fetch models: {e}")
    model = st.selectbox("Gemini multimodal model", ss.models)
    temperature = st.slider("Temperature", 0.0, 1.5, 0.4, 0.1)

    st.divider()
    st.header("⚙️ Optional overrides")
    fps = st.number_input("Frame rate (fps)", 12, 60, 24)
    render_engine = st.selectbox("Render engine", ["EEVEE", "Cycles"])

# --------------------------------------------------------------------------- #
# Prompts
# --------------------------------------------------------------------------- #
ANALYZE_SYSTEM = """You read animation storyboards and extract every detail needed to build the scene in 3D.
Read ALL text and visuals in the image(s): title, panel numbers, timecodes, panel titles, captions, characters,
outfits, props, environment, lighting, camera angles and actions. Infer anything not written from the visuals.
Convert timecodes like "0:35 – 0:55" to seconds. Return ONLY valid JSON."""

ANALYZE_SCHEMA = """{
  "project_title": str,
  "visual_style": str,
  "overall_environment": str,
  "total_duration_s": number,
  "characters": [{
    "name": str,
    "species_or_type": str,
    "appearance": str,
    "outfit_and_accessories": str,
    "colors": str,
    "approx_height_m": number,
    "personality": str
  }],
  "segments": [{
    "number": int,
    "title": str,
    "start_s": number,
    "end_s": number,
    "caption_text": str,
    "visual_description": str,
    "characters_present": [str],
    "actions": [str],
    "camera": str,
    "props_and_effects": [str],
    "mood_and_lighting": str
  }]
}"""

GENERATE_SYSTEM = """You are a technical director who converts storyboard segments into precise, Blender-ready
scene specifications. Another AI (Claude, using a Blender MCP connector) will read your output and write a
Blender Python (bpy) script, so be concrete and unambiguous.

Rules:
- Blender conventions: right-handed, Z-up, metres, rotations in degrees (Euler XYZ), camera looks down its local -Z.
- Every character needs import instructions (how to obtain/import the asset) AND exact placement
  (location, rotation, scale) plus a starting pose.
- Fill in every detail the storyboard leaves out (positions, distances, lens, easing, timing) with sensible values.
- Times are in seconds from the start of the whole animation; also give frame numbers at the stated fps.
- Animation is a list of keyframes with explicit values; avoid vague words.
- Maintain continuity with the previous segment's end state when provided.
- Return ONLY valid JSON, no markdown fences."""

GENERATE_SCHEMA = """{
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

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def to_parts(files):
    return [
        types.Part.from_bytes(data=f.getvalue(), mime_type=f.type or "image/png")
        for f in files or []
    ]


def parse_json(text):
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    return json.loads(text)


def call_gemini(client, system, contents):
    cfg = types.GenerateContentConfig(
        system_instruction=system,
        temperature=temperature,
        response_mime_type="application/json",
    )
    err = None
    for attempt in range(3):
        try:
            return client.models.generate_content(model=model, contents=contents, config=cfg).text
        except Exception as e:  # noqa: BLE001
            err = e
            time.sleep(2 * (attempt + 1))
    raise err


def run_pipeline(story_files, ref_files, notes):
    client = genai.Client(api_key=ss.api_key)
    story_parts = to_parts(story_files)
    ref_parts = to_parts(ref_files)
    notes_txt = f"\nExtra notes from the user: {notes}" if notes.strip() else ""

    # ---- Stage 1: read the storyboard ------------------------------------ #
    status = st.status("Reading storyboard…", expanded=True)
    contents = ["Storyboard image(s):", *story_parts]
    if ref_parts:
        contents += ["Extra character/environment reference images:", *ref_parts]
    contents.append(f"Extract everything into this JSON shape:\n{ANALYZE_SCHEMA}{notes_txt}")
    raw = call_gemini(client, ANALYZE_SYSTEM, contents)
    try:
        analysis = parse_json(raw)
    except json.JSONDecodeError:
        status.update(label="Could not parse the storyboard analysis", state="error")
        st.code(raw)
        return
    ss.analysis = analysis
    segments = analysis.get("segments", [])
    status.write(
        f"Found {len(analysis.get('characters', []))} characters and {len(segments)} segments."
    )

    # ---- Stage 2: one Blender spec per segment --------------------------- #
    ss.results = []
    prev_end = ""
    progress = st.progress(0.0)
    for i, seg in enumerate(segments):
        status.write(f"Generating segment {i + 1}/{len(segments)}: {seg.get('title', '')}")
        contents = ["Storyboard image(s):", *story_parts]
        if ref_parts:
            contents += ["Extra reference images:", *ref_parts]
        contents.append(
            f"""Project: {analysis.get('project_title', '')}
Visual style: {analysis.get('visual_style', '')}
Overall environment: {analysis.get('overall_environment', '')}
FPS: {fps} | Render engine: {render_engine}

Characters:
{json.dumps(analysis.get('characters', []), indent=2)}

Segment to build now ({i + 1} of {len(segments)}):
{json.dumps(seg, indent=2)}

Previous segment end state: {prev_end or 'n/a (first segment)'}{notes_txt}

Return JSON matching this shape exactly:
{GENERATE_SCHEMA}"""
        )
        raw = call_gemini(client, GENERATE_SYSTEM, contents)
        try:
            data = parse_json(raw)
            prev_end = data.get("end_state", "") if isinstance(data, dict) else ""
        except json.JSONDecodeError:
            data = {"_raw": raw, "_error": "Model did not return valid JSON"}
        ss.results.append((seg, data))
        progress.progress((i + 1) / len(segments))

    status.update(label="Done", state="complete", expanded=False)
    build_final_prompt()


def build_final_prompt():
    a = ss.analysis or {}
    L = ["# TASK: Build a Blender animation via the Blender MCP connector\n"]
    L.append(
        "Using the Blender connector, write and execute a Python (bpy) script that builds the scene "
        "below. Work segment by segment so I can review each one.\n"
    )
    L.append("## Requirements")
    L.append(f"- FPS: {fps}  |  Render engine: {render_engine}")
    L.append("- Units: metres, Z-up. Rotations are in degrees (Euler XYZ).")
    if a.get("visual_style"):
        L.append(f"- Style: {a['visual_style']}")
    L.append(
        "- Import each character using its `import_strategy` (local file, Sketchfab, Poly Haven, "
        "Hyper3D/Hunyuan3D, or a primitive stand-in if unavailable) and apply the exact placement."
    )
    L.append("- Use keyframes (`keyframe_insert`) with the times/frames given; set the scene frame range to cover all segments.")
    L.append("- Name every object clearly so later segments can reference them.")
    L.append("- After each segment, report what was created and any assumptions or fallbacks.\n")

    if a.get("project_title"):
        L.append(f"## Project: {a['project_title']}")
    if a.get("overall_environment"):
        L.append(f"Environment: {a['overall_environment']}\n")

    L.append("## Characters")
    for c in a.get("characters", []):
        L.append(f"- **{c.get('name', 'Unnamed')}** ({c.get('species_or_type', '')}): {c.get('appearance', '')}")
        if c.get("outfit_and_accessories"):
            L.append(f"  - Outfit: {c['outfit_and_accessories']}")
        if c.get("colors"):
            L.append(f"  - Colors: {c['colors']}")
        if c.get("approx_height_m"):
            L.append(f"  - Approx height: {c['approx_height_m']} m")

    L.append("\n## Segments")
    for i, (seg, data) in enumerate(ss.results, 1):
        L.append(f"\n### Segment {i}: {seg.get('title', 'Untitled')} ({seg.get('start_s')}s–{seg.get('end_s')}s)")
        if "_raw" in data:
            L.append(data["_raw"])
            continue
        if data.get("summary"):
            L.append(data["summary"])
        if data.get("claude_instructions"):
            L.append(f"\n**Instructions:** {data['claude_instructions']}")
        L.append("\n```json")
        L.append(json.dumps(data, indent=2, ensure_ascii=False))
        L.append("```")

    L.append("\n## Deliverable")
    L.append(
        "Start with Segment 1: generate the bpy script, execute it through the Blender connector, "
        "and confirm the result before continuing."
    )
    ss.final_prompt = "\n".join(L)


# --------------------------------------------------------------------------- #
# Main UI
# --------------------------------------------------------------------------- #
st.title("🎬 Storyboard → Blender Prompts")
st.caption("Upload your storyboard. Gemini reads it and generates everything else.")

story_files = st.file_uploader(
    "1 · Storyboard image(s)",
    type=["png", "jpg", "jpeg", "webp"],
    accept_multiple_files=True,
)
if story_files:
    st.image([f.getvalue() for f in story_files], width=420)

with st.expander("Optional: extra references and notes"):
    ref_files = st.file_uploader(
        "Character / environment reference images",
        type=["png", "jpg", "jpeg", "webp"],
        accept_multiple_files=True,
        key="refs",
    )
    notes = st.text_area("Extra notes for the model", placeholder="Anything the storyboard doesn't show.")

if not ss.api_key:
    st.info("Save your Gemini API key in the sidebar first.")

if st.button("✨ Analyze & generate", type="primary", disabled=not (ss.api_key and story_files)):
    try:
        run_pipeline(story_files, ref_files, notes)
    except Exception as e:  # noqa: BLE001
        st.error(f"Generation failed: {e}")

# --------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------- #
if ss.analysis:
    with st.expander("What Gemini read from the storyboard", expanded=False):
        st.json(ss.analysis, expanded=1)

if ss.results:
    with st.expander("Per-segment Blender specs", expanded=False):
        for i, (seg, data) in enumerate(ss.results, 1):
            st.markdown(f"**Segment {i}: {seg.get('title', '')}**")
            if "_raw" in data:
                st.warning(data["_error"])
                st.code(data["_raw"])
            else:
                st.json(data, expanded=False)

if ss.final_prompt:
    st.subheader("2 · Copy into Claude")
    st.caption("Use the copy icon at the top-right of the block, or download the file.")
    st.code(ss.final_prompt, language="markdown", wrap_lines=True)
    st.download_button(
        "⬇️ Download as .md",
        ss.final_prompt,
        file_name="blender_prompt_for_claude.md",
        mime="text/markdown",
    )

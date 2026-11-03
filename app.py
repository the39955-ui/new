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
ss.setdefault("segment_prompts", [])

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


def generate_json(client, system, contents, tries=3):
    """Call Gemini and return (parsed_json, raw_text). Retries if JSON is invalid."""
    raw = ""
    for _ in range(tries):
        raw = call_gemini(client, system, contents)
        try:
            return parse_json(raw), raw
        except json.JSONDecodeError:
            contents = [*contents, "Your previous reply was not valid JSON. Return ONLY strict JSON: "
                        "double-quoted keys, no comments, no trailing commas, no extra keys."]
    return None, raw


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
    analysis, raw = generate_json(client, ANALYZE_SYSTEM, contents)
    if analysis is None:
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
        data, raw = generate_json(client, GENERATE_SYSTEM, contents)
        if data is None:
            data = {"_raw": raw, "_error": "Model did not return valid JSON"}
        else:
            prev_end = data.get("end_state", "")
        ss.results.append((seg, data))
        progress.progress((i + 1) / len(segments))

    status.update(label="Done", state="complete", expanded=False)
    build_segment_prompts()


def build_segment_prompts():
    a = ss.analysis or {}
    chars = a.get("characters", [])
    n = len(ss.results)
    out = []

    for i, (seg, data) in enumerate(ss.results, 1):
        title = seg.get("title", "Untitled")
        ok = isinstance(data, dict) and "_raw" not in data
        tr = data.get("time_range", {}) if ok else {}
        start_f = tr.get("start_frame", int(round((seg.get("start_s") or 0) * fps)))
        end_f = tr.get("end_frame", int(round((seg.get("end_s") or 0) * fps)))
        prev_end = ""
        if i > 1 and isinstance(ss.results[i - 2][1], dict):
            prev_end = ss.results[i - 2][1].get("end_state", "")

        L = [f"# TASK: Build Segment {i} of {n}: {title} ({seg.get('start_s')}s to {seg.get('end_s')}s)\n"]
        L.append(
            "Use the Blender MCP connector to write and execute a Python (bpy) script that builds "
            "THIS segment only.\n"
        )

        L.append("## Scene state")
        if i == 1:
            L.append(
                f"- This is the first segment. Use the open Blender file (clear default objects). "
                f"Set fps to {fps}, render engine to {render_engine}, frame start 0, frame end {end_f}."
            )
        else:
            L.append(
                f"- Segments 1 to {i - 1} are already built in the open Blender file. "
                "Do NOT delete or rebuild them. Reuse existing objects by name where they exist."
            )
            L.append(f"- Add this segment on frames {start_f} to {end_f} and extend the scene frame end to {end_f}.")
            if prev_end:
                L.append(f"- Previous segment ended with: {prev_end}")
        L.append("")

        L.append("## Requirements")
        L.append(f"- FPS: {fps}  |  Render engine: {render_engine}")
        L.append("- Units: metres, Z-up. Rotations are in degrees (Euler XYZ).")
        if a.get("visual_style"):
            L.append(f"- Style: {a['visual_style']}")
        L.append(
            "- Import each character using its `import_strategy` (local file, Sketchfab, Poly Haven, "
            "Hyper3D/Hunyuan3D, or a primitive stand-in if unavailable) and apply the exact placement."
        )
        L.append("- Use keyframes (`keyframe_insert`) with the exact times and frames given.")
        L.append("- Name every object clearly so later segments can reference them.\n")

        if a.get("project_title"):
            L.append(f"## Project: {a['project_title']}")
        if a.get("overall_environment"):
            L.append(f"Environment: {a['overall_environment']}\n")

        names = {c.get("name") for c in data.get("characters", [])} if ok else set()
        present = [c for c in chars if c.get("name") in names] or chars
        L.append("## Characters in this segment")
        for c in present:
            L.append(f"- **{c.get('name', 'Unnamed')}** ({c.get('species_or_type', '')}): {c.get('appearance', '')}")
            if c.get("outfit_and_accessories"):
                L.append(f"  - Outfit: {c['outfit_and_accessories']}")
            if c.get("colors"):
                L.append(f"  - Colors: {c['colors']}")
            if c.get("approx_height_m"):
                L.append(f"  - Approx height: {c['approx_height_m']} m")

        L.append(f"\n## Segment {i} spec")
        if ok:
            if data.get("summary"):
                L.append(data["summary"])
            if data.get("claude_instructions"):
                L.append(f"\n**Instructions:** {data['claude_instructions']}")
            L.append("\n```json")
            L.append(json.dumps(data, indent=2, ensure_ascii=False))
            L.append("```")
        else:
            L.append(data.get("_raw", ""))

        L.append("\n## Deliverable")
        L.append(
            "Generate the bpy script, execute it through the Blender connector, and report what was "
            "created plus any assumptions or fallbacks. Then stop and wait for the next segment."
        )
        out.append({"title": title, "text": "\n".join(L)})

    ss.segment_prompts = out
    ss.final_prompt = "\n\n---\n\n".join(p["text"] for p in out)


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

if ss.segment_prompts:
    st.subheader("2 · Copy one segment at a time into Claude")
    st.caption("Each tab is a complete, standalone prompt. Copy Segment 1, paste into Claude, wait for it to finish, then do Segment 2.")
    tabs = st.tabs([f"{i}. {p['title']}" for i, p in enumerate(ss.segment_prompts, 1)])
    for i, (tab, p) in enumerate(zip(tabs, ss.segment_prompts), 1):
        with tab:
            st.code(p["text"], language="markdown", wrap_lines=True)
            st.download_button(
                f"⬇️ Download segment {i} (.md)",
                p["text"],
                file_name=f"segment_{i}_{p['title'].lower().replace(' ', '_')}.md",
                mime="text/markdown",
                key=f"dl_{i}",
            )
    with st.expander("All segments in one file"):
        st.download_button(
            "⬇️ Download all segments (.md)",
            ss.final_prompt,
            file_name="blender_prompts_all_segments.md",
            mime="text/markdown",
            key="dl_all",
        )

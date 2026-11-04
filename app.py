"""
Storyboard -> Blender Prompt Generator (asset-aware, single Gemini call)

Upload a storyboard, list the assets you've ALREADY imported in Blender (Bobo, Zia, Bus...),
and Gemini produces one standalone prompt per segment. Paste each into Claude (Blender MCP).
Claude is told to use your assets as-is and to match the scene's theme to them.

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
ss.setdefault("plan", None)
ss.setdefault("assets_text", "")
ss.setdefault("segment_prompts", [])
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
    temperature = st.slider("Temperature", 0.0, 1.5, 0.3, 0.1)

    st.divider()
    st.header("⚙️ Scene overrides")
    fps = st.number_input("Frame rate (fps)", 12, 60, 24)
    render_engine = st.selectbox("Render engine", ["EEVEE", "Cycles"])

# --------------------------------------------------------------------------- #
# Prompts for Gemini (ONE call does everything)
# --------------------------------------------------------------------------- #
SYSTEM = """You are a technical director converting an animation storyboard into Blender-ready scene specs.
Another AI (Claude, with a Blender MCP connector) will read your output and write bpy scripts.

STORYBOARD READING
- Read ALL text and visuals: title, panel numbers, timecodes ("0:35 - 0:55" -> seconds), captions, characters,
  props, environment, camera angles, actions.

EXISTING ASSETS (most important)
- The user may list assets ALREADY imported in their Blender scene (names as shown in the Outliner).
- Map every storyboard character / vehicle / prop to the closest listed asset and put its EXACT listed name
  in "existing_asset". If nothing matches, set "existing_asset" to null.
- Existing assets are READ-ONLY. For them give ONLY object-level location and rotation_deg and animation of
  location / rotation (or existing pose bones / shape keys if the user says they exist). NEVER give scale,
  materials, colours, redesign or "appearance" changes for them. Do not describe how they look.
- The scene's look must follow the user's assets, not generic defaults: in "visual_theme_notes" say that the
  environment, props, lighting and world should be built to match the style of the existing assets
  (inspect their materials and palette). Use the storyboard only for layout, action and mood.
- For things with no existing asset, give a short appearance and an import_strategy
  (Sketchfab / Poly Haven / Hyper3D / primitive stand-in).

FACIAL PERFORMANCE (required for every character that appears on screen)
- Analyse each segment's mood, action and caption, then give each character an "emotion_arc" and an
  "expressions" timeline that fits the scene (e.g. surprise, laughter, delight, cheeky, sleepy, scared).
- Each expression entry: time_s, frame, emotion, intensity (0 to 1), and short details for eyes, brows
  and mouth. Include eye direction (what the character looks at), blinks, and mouth motion for talking,
  singing, laughing, chewing, licking or sneezing. Use 2 to 6 entries per character per segment.
- Expressions must start a few frames BEFORE the action they react to (anticipation) and hold after it.
- These are performance targets only. Never redesign a character's face. For existing assets, the facial
  animation must use whatever controls the asset already has (shape keys, face bones, custom properties).

SPEC RULES
- Blender: right-handed, Z-up, metres, degrees (Euler XYZ), camera looks down local -Z.
- Coordinates are layout targets; the real asset sizes are unknown to you, so keep relationships
  (distances, facing, framing) rather than absolute precision.
- Define the persistent environment ONCE at top level. Per segment list only "environment_changes".
- Times are seconds from the start of the whole animation, plus frame numbers at the given fps.
- Be concise: no filler, no repeated information, keyframes only where motion happens.
- Return ONLY strict JSON: double-quoted keys, no comments, no trailing commas, no extra keys."""

SCHEMA = """{
  "project_title": str,
  "visual_theme_notes": str,
  "total_duration_s": number,
  "characters": [{"name": str, "kind": "character"|"vehicle"|"prop", "existing_asset": str|null, "appearance": str|null, "import_strategy": str|null}],
  "environment": {"setting": str, "objects": [{"name": str, "type": str, "location": [x,y,z], "rotation_deg": [x,y,z], "scale": [x,y,z]}], "lighting": str, "world_background": str},
  "segments": [{
    "number": int,
    "title": str,
    "time_range": {"start_s": number, "end_s": number, "start_frame": int, "end_frame": int},
    "summary": str,
    "environment_changes": [str],
    "characters": [{"name": str, "placement": {"location": [x,y,z], "rotation_deg": [x,y,z]}, "pose_note": str, "emotion_arc": str, "expressions": [{"time_s": number, "frame": int, "emotion": str, "intensity": number, "eyes": str, "brows": str, "mouth": str}]}],
    "camera": {"shot_type": str, "location": [x,y,z], "rotation_deg": [x,y,z], "focal_length_mm": number, "dof": {"enabled": bool, "focus_target": str, "f_stop": number}, "movement": str},
    "animation": [{"target": str, "action": str, "keyframes": [{"time_s": number, "frame": int, "property": str, "value": str}], "easing": str}],
    "claude_instructions": str,
    "end_state": str
  }]
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
    return json.loads(re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip()))


def call_gemini(client, contents):
    cfg = types.GenerateContentConfig(
        system_instruction=SYSTEM,
        temperature=temperature,
        response_mime_type="application/json",
    )
    return client.models.generate_content(model=model, contents=contents, config=cfg).text


def generate_plan(client, contents):
    """One call normally. A single retry only if the JSON is broken or the request errors."""
    raw = ""
    for attempt in range(2):
        try:
            raw = call_gemini(client, contents)
            return parse_json(raw), raw
        except json.JSONDecodeError:
            contents = [*contents, "Your previous reply was not valid JSON. Return strict JSON only."]
        except Exception:  # noqa: BLE001
            if attempt == 1:
                raise
            time.sleep(3)
    return None, raw


def run_pipeline(story_files, asset_text, asset_files, notes):
    client = genai.Client(api_key=ss.api_key)
    contents = ["Storyboard image(s):", *to_parts(story_files)]
    if asset_files:
        contents += ["Screenshots of the assets already imported in my Blender scene:", *to_parts(asset_files)]

    contents.append(
        f"""FPS: {fps} | Render engine: {render_engine}

Assets ALREADY imported in my Blender scene (exact Outliner names):
{asset_text.strip() or 'None. Nothing is imported yet; suggest import strategies.'}
{('Extra notes: ' + notes.strip()) if notes.strip() else ''}

Return JSON in exactly this shape:
{SCHEMA}"""
    )

    with st.spinner("Gemini is reading the storyboard and planning every segment (1 call)…"):
        plan, raw = generate_plan(client, contents)

    if plan is None:
        st.error("Gemini returned invalid JSON twice. Try again or switch to a stronger model.")
        st.code(raw)
        return
    ss.plan = plan
    ss.assets_text = asset_text
    build_segment_prompts()


# --------------------------------------------------------------------------- #
# Per-segment prompt for Claude
# --------------------------------------------------------------------------- #
ASSET_RULES = """## My assets already in the Blender scene (READ-ONLY)
These objects are ALREADY imported and finished. Treat their design as final:
- Do NOT import, duplicate, remodel, retexture, recolour, re-material, add or remove modifiers,
  edit meshes / rigs / shape keys, change origins, or change scale or dimensions.
- Only change object-level location and rotation (and use existing pose bones / shape keys / actions if
  they exist). Never apply or bake transforms in a way that alters the mesh.
- Before building, inspect the scene ONCE (names, types, dimensions, bounding boxes, forward axis,
  armature bones, shape keys, materials). Use the REAL measured sizes, not guesses. Put each asset's lowest
  point on the ground, face it correctly, and treat the coordinates in the spec as layout targets that you
  adjust to the real sizes while keeping the same distances, facing and framing.
- THEME: build the environment, props, lighting and world to match the look of these assets
  (their materials, palette, shading style such as toon vs PBR, outline, level of detail). The storyboard
  gives layout, action and mood only.
- If a needed asset is not in my scene, use a simple stand-in or the import strategy given, and say so.

My asset list:
{assets}
"""



FACE_RULES = """## Facial expressions and performance (required)
Animate every on-screen character's face so it matches the scene. Use the `expressions` timeline in the spec
(emotion, intensity 0 to 1, eyes, brows, mouth) as the performance plan.
- In your single scene inspection, also list each character's facial controls: shape keys on the face/head/eye
  meshes, face or jaw or eye or brow bones, custom properties, and drivers. Match the spec's emotions to the
  CLOSEST controls that already exist (smile, brow raise, eyes wide, blink, jaw open, mouth shapes, eye look).
- Animate only VALUES with keyframes (shape key values, bone rotations, custom property values). Never create,
  rename, edit or delete shape keys, bones, drivers or materials on my assets.
- Start each expression 3 to 6 frames before the action it reacts to, ease in, hold, then settle back to neutral.
- Add natural life: a blink every 3 to 5 seconds, eyes looking at the thing they react to, small head
  tilts, and mouth motion for talking, singing, laughing, chewing, licking or sneezing (oscillate the jaw or
  mouth shapes if there are no speech shapes).
- If a character has no facial controls at all, animate head and eye bones or object-level head motion only,
  and tell me exactly what was missing.
- In your report, list per character which controls you used for expressions.
"""

def vec_safe(d, key, default):
    v = d.get(key)
    return v if v is not None else default


def build_segment_prompts():
    plan = ss.plan or {}
    segs = plan.get("segments", [])
    n = len(segs)
    cmap = {c.get("name"): c for c in plan.get("characters", [])}
    env = plan.get("environment", {})
    env_names = [o.get("name") for o in env.get("objects", []) if o.get("name")]
    assets = ss.assets_text.strip()
    out = []

    for i, seg in enumerate(segs, 1):
        title = seg.get("title", "Untitled")
        tr = seg.get("time_range", {})
        s_s, e_s = tr.get("start_s"), tr.get("end_s")
        start_f = tr.get("start_frame", int(round((s_s or 0) * fps)))
        end_f = tr.get("end_frame", int(round((e_s or 0) * fps)))
        prev_end = segs[i - 2].get("end_state", "") if i > 1 else ""

        L = [f"# TASK: Build Segment {i} of {n}: {title} ({s_s}s to {e_s}s)\n"]
        L.append(
            "Use the Blender MCP connector to build THIS segment only. Keep tool calls minimal: "
            "one scene inspection, then ONE bpy script that builds and keyframes everything, "
            "then one quick check.\n"
        )

        L.append("## Scene state")
        if i == 1:
            L.append(
                f"- First segment. Use the open Blender file and KEEP my imported assets untouched. "
                f"Remove only default cube/light/camera if present. FPS {fps}, engine {render_engine}, "
                f"frame start 0, frame end {end_f}."
            )
        else:
            L.append(
                f"- Segments 1 to {i - 1} are already built in the open file. Do NOT delete or rebuild them."
                " Reuse existing objects by name."
            )
            L.append(f"- Add this segment on frames {start_f} to {end_f} and extend the frame end to {end_f}.")
            if prev_end:
                L.append(f"- Previous segment ended with: {prev_end}")
        L.append("")

        if assets:
            L.append(ASSET_RULES.format(assets=assets))

        L.append(FACE_RULES)

        L.append("## Requirements")
        L.append(f"- FPS {fps} | Engine {render_engine} | Units metres, Z-up | Rotations in degrees (Euler XYZ)")
        if plan.get("visual_theme_notes"):
            L.append(f"- Theme: {plan['visual_theme_notes']}")
        L.append("- Use `keyframe_insert` with the exact frames given. Name every object clearly.\n")

        if plan.get("project_title"):
            L.append(f"## Project: {plan['project_title']}")

        L.append("## Who is in this segment")
        for c in seg.get("characters", []):
            info = cmap.get(c.get("name"), {})
            if info.get("existing_asset"):
                L.append(f"- **{c.get('name')}** → my existing asset `{info['existing_asset']}` (do not modify it)")
            else:
                desc = info.get("appearance") or ""
                how = info.get("import_strategy") or "simple stand-in"
                L.append(f"- **{c.get('name')}** → NOT in my scene. {desc} Obtain via: {how}")

        L.append("\n## Environment")
        if i == 1:
            L.append("Build this persistent environment now (match my assets' style):")
            L.append("```json\n" + json.dumps(env, indent=2, ensure_ascii=False) + "\n```")
        else:
            if env_names:
                L.append("Already built in earlier segments: " + ", ".join(env_names) + ".")
            for ch in seg.get("environment_changes", []):
                L.append(f"- Change: {ch}")

        L.append(f"\n## Segment {i} spec")
        if seg.get("summary"):
            L.append(seg["summary"])
        if seg.get("claude_instructions"):
            L.append(f"\n**Instructions:** {seg['claude_instructions']}")
        spec = {k: v for k, v in seg.items() if k not in ("claude_instructions", "summary", "environment_changes")}
        L.append("\n```json\n" + json.dumps(spec, indent=2, ensure_ascii=False) + "\n```")

        L.append("\n## Deliverable")
        L.append(
            "Run the script through the connector, then report in a few lines what was created or reused, "
            "which facial controls drove each character's expressions, and any assumptions or fallbacks. "
            "Then stop and wait for the next segment."
        )
        out.append({"title": title, "text": "\n".join(L)})

    ss.segment_prompts = out
    ss.final_prompt = "\n\n---\n\n".join(p["text"] for p in out)


# --------------------------------------------------------------------------- #
# Main UI
# --------------------------------------------------------------------------- #
st.title("🎬 Storyboard → Blender Prompts")
st.caption("Upload the storyboard, list the assets you already imported, get one prompt per segment.")

story_files = st.file_uploader(
    "1 · Storyboard image(s)",
    type=["png", "jpg", "jpeg", "webp"],
    accept_multiple_files=True,
)
if story_files:
    st.image([f.getvalue() for f in story_files], width=420)

asset_text = st.text_area(
    "2 · Assets already imported in my Blender scene",
    placeholder="One per line, exact Outliner name, with what it is. Example:\nBobo (boy character, rigged)\nZia (girl character, rigged)\nBus (school bus vehicle)",
    height=120,
    help="Claude will use these exactly as they are and match the scene's theme to them.",
)

with st.expander("Optional: asset screenshots and notes"):
    asset_files = st.file_uploader(
        "Screenshots of your imported assets (helps Gemini match the theme)",
        type=["png", "jpg", "jpeg", "webp"],
        accept_multiple_files=True,
        key="assets_img",
    )
    notes = st.text_area("Extra notes", placeholder="Anything the storyboard doesn't show.")

if not ss.api_key:
    st.info("Save your Gemini API key in the sidebar first.")

if st.button("✨ Generate (1 Gemini call)", type="primary", disabled=not (ss.api_key and story_files)):
    try:
        run_pipeline(story_files, asset_text, asset_files, notes)
    except Exception as e:  # noqa: BLE001
        st.error(f"Generation failed: {e}")

if ss.plan:
    with st.expander("What Gemini planned (raw JSON)", expanded=False):
        st.json(ss.plan, expanded=1)

if ss.segment_prompts:
    st.subheader("3 · Copy one segment at a time into Claude")
    st.caption("Each tab is a standalone prompt. Copy Segment 1, paste into Claude, wait for it to finish, then do Segment 2.")
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

import json
import re
from pathlib import Path


def load_profile():
    info = json.loads(Path("../info.json").read_text(encoding="utf-8"))
    return str(info.get("wind", {}).get("profile", "legacy_fixed")).strip().lower()


def replace_patch_type(boundary_text, patch_name, new_type):
    pattern = rf"(^[ \t]*{re.escape(patch_name)}\s*\{{[^{{}}]*?\btype\s+)(\w+)(\s*;)"

    def repl(match):
        return f"{match.group(1)}{new_type}{match.group(3)}"

    updated, count = re.subn(pattern, repl, boundary_text, flags=re.MULTILINE)
    if count == 0:
        raise RuntimeError(f"Failed to locate patch '{patch_name}' in constant/polyMesh/boundary")
    return updated


def replace_field_boundary_type(field_text, patch_name, old_type, new_type):
    pattern = rf"(^[ \t]*{re.escape(patch_name)}\s*\{{[^{{}}]*?\btype\s+){re.escape(old_type)}(\s*;)"
    return re.sub(pattern, rf"\1{new_type}\2", field_text, flags=re.MULTILINE)


def ensure_nut_top_entry(field_text):
    if re.search(r"(?m)^[ \t]*top\s*\{", field_text):
        return field_text

    include_line = '    #includeEtc "caseDicts/setConstraintTypes"'
    top_block = """    top
    {
        type            calculated;
        value           uniform 0;
    }

"""
    if include_line not in field_text:
        raise RuntimeError("Failed to locate setConstraintTypes include in 0/nut")
    return field_text.replace(include_line, top_block + include_line)


def main():
    profile = load_profile()
    boundary_path = Path("constant/polyMesh/boundary")
    text = boundary_path.read_text(encoding="utf-8")
    # The historical modifyBoundary binary matches exact indentation and can
    # return success without changing a patch. Verify/write the actual contract.
    updated = text
    patch_types = {"inlet": "patch", "outlet": "patch", "front": "symmetry",
                   "back": "symmetry", "bot": "wall", "top": "patch" if profile == "abl_log" else "symmetry"}
    for patch, patch_type in patch_types.items():
        updated = replace_patch_type(updated, patch, patch_type)
    boundary_path.write_text(updated, encoding="utf-8")
    print(f"[INFO] Verified six mesh patch types for profile={profile}")
    if profile != "abl_log":
        return

    zero_dir = Path("0")
    updated_files = []
    for field_path in zero_dir.iterdir():
        if not field_path.is_file():
            continue
        text = field_path.read_text(encoding="utf-8")
        new_text = replace_field_boundary_type(text, "top", "symmetry", "zeroGradient")
        if field_path.name == "nut":
            new_text = ensure_nut_top_entry(new_text)
        if new_text != text:
            field_path.write_text(new_text, encoding="utf-8")
            updated_files.append(field_path.name)

    if updated_files:
        print("[INFO] Updated top boundary type symmetry->zeroGradient in:", ", ".join(sorted(updated_files)))


if __name__ == "__main__":
    main()

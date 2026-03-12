#!/usr/bin/env python3
"""
Pre-commit hook to verify that bindtool header hashes are up-to-date.

This script prevents the common GNU Radio OOT footgun where C++ headers
are modified but the checked-in pybind11 binding files are not regenerated.
It works by comparing the BINDTOOL_HEADER_FILE_HASH(...) recorded in each
binding file against the actual MD5 hash of the referenced header.
"""

import hashlib
import re
import sys
from pathlib import Path


def find_binding_files(root_dir):
    """Find all binding translation units under python/**/bindings/*_python.cc."""
    root = Path(root_dir)
    python_dir = root / "python"
    if not python_dir.exists():
        return []

    binding_files = list(python_dir.glob("**/bindings/*_python.cc"))
    return binding_files


def extract_bindtool_metadata(binding_file):
    """
    Extract BINDTOOL_HEADER_FILE and BINDTOOL_HEADER_FILE_HASH from a binding file.

    The metadata is embedded in C-style comment blocks (typically near the top of the
    binding file), e.g., /* BINDTOOL_HEADER_FILE(variable_delay.h) */

    Returns:
        tuple: (header_filename, recorded_hash) or (None, None) if not found
    """
    content = binding_file.read_text()

    # Match: /* BINDTOOL_HEADER_FILE(variable_delay.h) */
    header_match = re.search(r"/\*\s*BINDTOOL_HEADER_FILE\(([^)]+)\)\s*\*/", content)

    # Match: /* BINDTOOL_HEADER_FILE_HASH(85d26c6347dfbf983415cf6d4ab5ce52) */
    hash_match = re.search(
        r"/\*\s*BINDTOOL_HEADER_FILE_HASH\(([a-fA-F0-9]+)\)\s*\*/", content
    )

    if not header_match or not hash_match:
        return None, None

    header_filename = header_match.group(1).strip()
    recorded_hash = hash_match.group(1).strip().lower()

    return header_filename, recorded_hash


def find_header_file(root_dir, header_filename):
    """
    Locate a header file under include/gnuradio/**/<header_filename>.

    Returns:
        Path: The header file path, or None if not found or ambiguous
    """
    root = Path(root_dir)
    include_dir = root / "include" / "gnuradio"

    if not include_dir.exists():
        return None

    matches = list(include_dir.glob(f"**/{header_filename}"))

    if len(matches) == 0:
        return None
    elif len(matches) > 1:
        return None
    else:
        return matches[0]


def compute_file_hash(file_path):
    """
    Compute MD5 hash of a file's contents.

    Uses the same method as GNU Radio's BindingGenerator.fix_file_hash().
    """
    hasher = hashlib.md5()
    with open(file_path, "rb") as file_in:
        buf = file_in.read()
        hasher.update(buf)
    return hasher.hexdigest()


def check_binding_file(root_dir, binding_file):
    """
    Check if a binding file's header hash matches the current header.

    Returns:
        tuple: (success: bool, error_message: str or None)
    """
    header_filename, recorded_hash = extract_bindtool_metadata(binding_file)

    if header_filename is None or recorded_hash is None:
        return True, None

    header_path = find_header_file(root_dir, header_filename)

    if header_path is None:
        return False, (
            f"ERROR: Could not locate header '{header_filename}' "
            f"referenced by {binding_file.relative_to(root_dir)}\n"
            f"  Expected to find it under include/gnuradio/**/"
        )

    current_hash = compute_file_hash(header_path)

    if current_hash != recorded_hash:
        rel_binding = binding_file.relative_to(root_dir)
        rel_header = header_path.relative_to(root_dir)
        return False, (
            f"\n❌ BINDTOOL HASH MISMATCH\n"
            f"  Binding: {rel_binding}\n"
            f"  Header:  {rel_header}\n"
            f"  Recorded hash: {recorded_hash}\n"
            f"  Current hash:  {current_hash}\n"
            f"\n"
            f"  The C++ header has changed but the Python bindings were not updated.\n"
            f"  To fix this:\n"
            f"    1. Run gr_modtool bind [block_name] to regenerate bindings, OR\n"
            f"    2. Manually update the BINDTOOL_HEADER_FILE_HASH in {rel_binding}\n"
        )

    return True, None


def main():
    """Main entry point for the pre-commit hook."""
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent

    binding_files = find_binding_files(repo_root)

    if not binding_files:
        print("✓ No binding files found to check")
        return 0

    all_success = True
    error_messages = []

    for binding_file in sorted(binding_files):
        success, error_msg = check_binding_file(repo_root, binding_file)
        if not success:
            all_success = False
            error_messages.append(error_msg)

    if all_success:
        print(
            f"✓ All {len(binding_files)} binding file(s) have up-to-date header hashes"
        )
        return 0
    else:
        for msg in error_messages:
            print(msg, file=sys.stderr)
        print(
            f"\n{len(error_messages)} binding file(s) out of sync with headers.",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())

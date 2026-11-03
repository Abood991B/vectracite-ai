"""Section-heading recognition shared by chunking and synthesis."""

from __future__ import annotations

import re

SECTION_HEADER_REGEX = re.compile(
    r"^(?:PROFESSIONAL\s+SUMMARY|SUMMARY|EXPERIENCE|WORK\s+EXPERIENCE|"
    r"TECHNICAL\s+SKILLS|SKILLS|PROJECTS|EDUCATION|CERTIFICATIONS|AWARDS|"
    r"LANGUAGES|PUBLICATIONS|OBJECTIVE|PROFILE|ITEM\s+\d+[A-Z]?\.?|NOTE\s+\d+|"
    r"PART\s+[IVX]+|[A-Z][A-Z0-9\s,&\-/]{3,45})$"
)
